"""Command-line interface.

Commands
--------
    process <image> [--mode demo|real] [--chain memory|anvil|testnet]
        Run the full pipeline end-to-end.
    verify <image>  [--mode demo|real] [--chain ...] [--tamper]
        Re-run search -> extraction -> fingerprint and verify against the chain
        (no new record is created).
    compare <imgA> <imgB>
        Compare two images supplied by the same user (similarity only).

Examples
--------
    python -m src.cli process ./data/input.jpg --mode demo
    python -m src.cli process ./data/input.jpg --mode real --chain anvil
    python -m src.cli verify ./data/input.jpg --mode demo --tamper
    python -m src.cli compare ./data/imgA.jpg ./data/imgB.jpg
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.blockchain.client import InMemoryBlockchainClient, Web3BlockchainClient
from src.face.service import FaceService, load_image_bgr
from src.models.schemas import PipelineConfig
from src.pipeline.pipeline import Pipeline, PipelineError
from src.provenance.extractor import ProvenanceExtractor
from src.search.fixture import DEFAULT_FIXTURE_PATH, load_fixture
from src.search.result_parser import MatchValidator
from src.search.reverse_image import FixtureSearchProvider, SerperLensSearchProvider
from src.search.web_search import HttpContentFetcher

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "input.jpg"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Component construction (dependency injection root)
# ---------------------------------------------------------------------------


def _tamper_fixture(fixture: dict) -> dict:
    """Return a deep copy of the fixture with content visibly altered."""
    tampered = json.loads(json.dumps(fixture))
    tampered["page"]["title"] = tampered["page"]["title"] + " [TAMPERED]"
    tampered["page"]["text"] = "TAMPERED CONTENT: " + tampered["page"]["text"]
    tampered["candidate"]["title"] = tampered["page"]["title"]
    return tampered


def build_components(mode: str, chain: str, tamper: bool = False):
    """Build components for the requested mode."""
    from src.search.fixture import FixtureContentFetcher

    face_service = FaceService()
    image_url = os.environ.get("SEARCH_IMAGE_URL") or None
    if mode == "demo":
        fixture = load_fixture(DEFAULT_FIXTURE_PATH)
        if tamper:
            fixture = _tamper_fixture(fixture)
        search_provider = FixtureSearchProvider(fixture=fixture)
        fetcher: object = FixtureContentFetcher(fixture=fixture)
    elif mode == "real":
        provider_name = os.environ.get("SEARCH_PROVIDER", "serper").lower()
        if provider_name != "serper":
            raise PipelineError(
                f"SEARCH_PROVIDER='{provider_name}' is not supported. "
                "Implemented providers: serper."
            )
        api_key = os.environ.get("SEARCH_API_KEY", "")
        if not api_key:
            raise PipelineError(
                "SEARCH_API_KEY is not set. Real mode needs a Serper API key "
                "(https://serper.dev). Use --mode demo for an offline demo."
            )
        search_provider = SerperLensSearchProvider(
            api_key=api_key,
            timeout=float(os.environ.get("SEARCH_REQUEST_TIMEOUT_SECONDS", "30")),
            max_candidates=int(os.environ.get("SEARCH_MAX_CANDIDATES", "10")),
            image_url=image_url,
        )
        fetcher = HttpContentFetcher(
            timeout=float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "20")),
            user_agent=os.environ.get("HTTP_USER_AGENT", ""),
            respect_robots=os.environ.get("ROBOTS_TXT_RESPECT", "true").lower()
            == "true",
            max_page_bytes=int(os.environ.get("MAX_PAGE_BYTES", "2097152")),
            max_image_bytes=int(os.environ.get("MAX_IMAGE_BYTES", "5242880")),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    validator = MatchValidator(fetcher)  # type: ignore[arg-type]
    extractor = ProvenanceExtractor(fetcher)  # type: ignore[arg-type]

    if chain == "memory":
        blockchain = InMemoryBlockchainClient()
    elif chain in ("anvil", "testnet"):
        rpc_url = os.environ.get("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:8545")
        private_key = os.environ.get("BLOCKCHAIN_PRIVATE_KEY") or None
        contract_address = os.environ.get("PROVENANCE_CONTRACT_ADDRESS") or None
        blockchain = Web3BlockchainClient(
            rpc_url=rpc_url,
            private_key=private_key,
            contract_address=contract_address,
            chain_name="local-anvil" if chain == "anvil" else "testnet",
        )
    else:
        raise ValueError(f"Unknown chain: {chain}")

    config = PipelineConfig(mode=mode)
    if mode == "real":
        config.image_url = image_url
    return face_service, search_provider, validator, extractor, blockchain, config


def ensure_demo_input(image: str | None) -> str:
    """Demo mode: fall back to the fixture image when the input is missing."""
    path = Path(image) if image else None
    if path is not None and path.exists():
        return str(path)
    fixture = load_fixture(DEFAULT_FIXTURE_PATH)
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    out = data_dir / "input.jpg"
    if not out.exists():
        import base64

        out.write_bytes(base64.b64decode(fixture["image_b64"]))
        logger.info("DEMO: wrote demo input image to %s", out)
    return str(out)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report(result) -> str:
    """Render the pipeline result in the assignment's sectioned format."""
    lines: list[str] = []
    face = result.face

    lines.append("[1] Face detection")
    lines.append(f"    Face detected: {'YES' if face.face_detected else 'NO'}")
    lines.append(f"    Faces: {face.face_count}")
    lines.append(f"    Privacy: {face.privacy_note}")
    lines.append("")

    if result.search is None:
        lines.append("    (pipeline stopped before search)")
        return "\n".join(lines)

    lines.append("[2] Reverse-image search")
    demo_label = " (DEMO MODE - local fixture, not a live search)" if result.search.demo else ""
    lines.append(f"    Provider: {result.search.provider}{demo_label}")
    lines.append(f"    Match found: {'YES' if result.search.match_found else 'NO'}")
    if result.search.match_found:
        lines.append(f"    Candidates: {len(result.search.candidates)}")
    lines.append("")

    if result.match is None:
        reason = result.no_match_reason or "No permitted matching public result found"
        lines.append("[3] Candidate validation")
        lines.append(f"    Match: NO ({reason})")
        lines.append("")
        lines.append("[FINAL] Pipeline stopped - no genuine public match.")
        return "\n".join(lines)

    lines.append("[3] Candidate validation")
    lines.append(f"    Match: YES")
    lines.append(f"    Match type: {result.match.match_type.value}")
    lines.append(f"    URL: {result.match.candidate.url}")
    lines.append(f"    Rationale: {result.match.rationale}")
    lines.append("")

    lines.append("[4] Fingerprint")
    lines.append(f"    SHA256: {result.fingerprint}")
    lines.append("")

    if result.chain is not None:
        lines.append("[5] Blockchain")
        sim = " (SIMULATED)" if result.chain.simulated else ""
        lines.append(f"    Blockchain: {result.chain.blockchain}{sim}")
        lines.append(f"    Contract: {result.chain.contract_address}")
        lines.append(f"    Transaction: {result.chain.transaction_hash}")
        lines.append(f"    Block: {result.chain.block_number}")
        lines.append("")

    if result.verification is not None:
        lines.append("[6] Verification")
        status = "VERIFIED" if result.verification.verified else "FAILED"
        lines.append(f"    Status: {status}")
        lines.append(f"    Reason: {result.verification.reason}")
        lines.append(f"    Calculated: {result.verification.calculated_hash}")
        lines.append(f"    On-chain: {result.verification.on_chain_hash or '(none)'}")
        lines.append(f"    Transaction: {result.verification.transaction_hash or '(none)'}")
        lines.append("")
        lines.append(f"[FINAL] VERIFICATION {status}")
    else:
        lines.append("[FINAL] Pipeline finished without verification step.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_process(args) -> int:
    image = ensure_demo_input(args.image) if args.mode == "demo" else args.image
    components = build_components(args.mode, args.chain, tamper=False)
    face_service, provider, validator, extractor, blockchain, config = components
    pipeline = Pipeline(face_service, provider, validator, extractor, blockchain, config)
    result = pipeline.run(image)
    print(format_report(result))
    return 0 if (result.completed or result.match is None) else 1


def cmd_verify(args) -> int:
    image = ensure_demo_input(args.image) if args.mode == "demo" else args.image
    components = build_components(args.mode, args.chain, tamper=args.tamper)
    face_service, provider, validator, extractor, blockchain, config = components
    config.record_on_chain = False
    pipeline = Pipeline(face_service, provider, validator, extractor, blockchain, config)
    result = pipeline.run(image)
    print(format_report(result))
    return 0 if result.verification and result.verification.verified else 1


def cmd_compare(args) -> int:
    face_service = FaceService()
    a = load_image_bgr(args.image_a)
    b = load_image_bgr(args.image_b)
    similarity = face_service.compare(a, b)
    print(json.dumps(similarity.model_dump(), indent=2))
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Face detection + web provenance + blockchain verification",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_process = sub.add_parser("process", help="Run the full pipeline")
    p_process.add_argument("image", nargs="?", default=str(DEFAULT_INPUT))
    p_process.add_argument("--mode", choices=["real", "demo"], default="demo")
    p_process.add_argument("--chain", choices=["memory", "anvil", "testnet"], default="memory")
    p_process.set_defaults(func=cmd_process)

    p_verify = sub.add_parser("verify", help="Verify provenance against the chain")
    p_verify.add_argument("image", nargs="?", default=str(DEFAULT_INPUT))
    p_verify.add_argument("--mode", choices=["real", "demo"], default="demo")
    p_verify.add_argument("--chain", choices=["memory", "anvil", "testnet"], default="memory")
    p_verify.add_argument(
        "--tamper",
        action="store_true",
        help="Demo mode only: alter the fixture content to demonstrate "
        "VERIFICATION FAILED",
    )
    p_verify.set_defaults(func=cmd_verify)

    p_compare = sub.add_parser(
        "compare", help="Compare two images supplied by the same user"
    )
    p_compare.add_argument("image_a")
    p_compare.add_argument("image_b")
    p_compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except (PipelineError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
