"""End-to-end offline demo.

Runs the complete pipeline with:
- the demo search provider (local fixture, explicitly labelled DEMO),
- the offline in-memory blockchain (explicitly labelled SIMULATED),
- the bundled public-domain sample face image.

It demonstrates BOTH required outcomes:
    1. successful verification  (PASS),
    2. tampered-content failure (FAIL).

For a real chain, run:
    python -m src.cli process ./data/input.jpg --mode real --chain anvil

Usage:
    python scripts/run_demo.py [--chain memory|anvil]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.blockchain.client import (  # noqa: E402
    InMemoryBlockchainClient,
    Web3BlockchainClient,
)
from src.cli import _tamper_fixture  # noqa: E402
from src.face.service import FaceService  # noqa: E402
from src.models.schemas import PipelineConfig  # noqa: E402
from src.pipeline.pipeline import Pipeline  # noqa: E402
from src.provenance.extractor import ProvenanceExtractor  # noqa: E402
from src.provenance.fingerprint import provenance_fingerprint  # noqa: E402
from src.search.fixture import (  # noqa: E402
    DEFAULT_FIXTURE_PATH,
    FixtureContentFetcher,
    load_fixture,
)
from src.search.result_parser import MatchValidator  # noqa: E402
from src.search.reverse_image import FixtureSearchProvider  # noqa: E402

logger = logging.getLogger(__name__)


def _demo_components(chain: str, fixture: dict):
    face_service = FaceService()
    provider = FixtureSearchProvider(fixture=fixture)
    fetcher = FixtureContentFetcher(fixture=fixture)
    validator = MatchValidator(fetcher)
    extractor = ProvenanceExtractor(fetcher)

    if chain == "anvil":
        blockchain = Web3BlockchainClient(
            rpc_url=os.environ.get("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:8545"),
            private_key=os.environ.get("BLOCKCHAIN_PRIVATE_KEY") or None,
            contract_address=os.environ.get("PROVENANCE_CONTRACT_ADDRESS") or None,
            chain_name="local-anvil",
        )
    else:
        blockchain = InMemoryBlockchainClient()
    return face_service, provider, validator, extractor, blockchain


def run_scenario(title: str, fixture: dict, chain: str, record: bool) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)

    face_service, provider, validator, extractor, blockchain = _demo_components(
        chain, fixture
    )
    config = PipelineConfig(mode="demo", record_on_chain=record)
    pipeline = Pipeline(face_service, provider, validator, extractor, blockchain, config)
    result = pipeline.run(str(PROJECT_ROOT / "data" / "sample_face.jpg"))

    print(f"[1] Face detection     -> detected={result.face.face_detected}, "
          f"faces={result.face.face_count}")
    if result.search:
        print(f"[2] Search             -> provider={result.search.provider}, "
              f"demo={result.search.demo}, match={result.search.match_found}")
    if result.match:
        print(f"[3] Validation         -> match_type={result.match.match_type.value}")
        print(f"    Rationale: {result.match.rationale}")
    if result.provenance:
        print(f"[4] Provenance record  -> url={result.provenance.source_url}")
        print(f"    title={result.provenance.title[:60]}...")
    if result.fingerprint:
        print(f"[5] Fingerprint        -> {result.fingerprint}")
    if result.chain:
        sim = " (SIMULATED)" if result.chain.simulated else ""
        print(f"[6] Blockchain         -> {result.chain.blockchain}{sim}")
        print(f"    tx={result.chain.transaction_hash} block={result.chain.block_number}")
    if result.verification:
        status = "PASS" if result.verification.verified else "FAIL"
        print(f"[7] Verification       -> {status}")
        print(f"    reason: {result.verification.reason}")
        print(f"    calculated: {result.verification.calculated_hash}")
        print(f"    on-chain:   {result.verification.on_chain_hash}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain", choices=["memory", "anvil"], default="memory")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    fixture = load_fixture(DEFAULT_FIXTURE_PATH)

    # Scenario 1: honest end-to-end -> VERIFIED
    run_scenario(
        "SCENARIO 1 - genuine demo pipeline (successful verification)",
        fixture,
        args.chain,
        record=True,
    )

    # Scenario 2: tampered content -> VERIFICATION FAILED
    tampered = _tamper_fixture(fixture)
    run_scenario(
        "SCENARIO 2 - tampered content (verification must FAIL)",
        tampered,
        args.chain,
        record=False,
    )

    # Explicit hash comparison for the tamper demonstration.
    from src.provenance.extractor import ProvenanceExtractor as PE

    fetcher_ok = FixtureContentFetcher(fixture=fixture)
    fetcher_bad = FixtureContentFetcher(fixture=tampered)
    rec_ok = PE(fetcher_ok).extract(
        source_url=fixture["candidate"]["url"],
        title_hint=fixture["candidate"]["title"],
        image_url=fixture["candidate"]["image_url"],
        search_provider="demo-fixture",
        match_type="EXACT_MATCH",
    )
    rec_bad = PE(fetcher_bad).extract(
        source_url=tampered["candidate"]["url"],
        title_hint=tampered["candidate"]["title"],
        image_url=tampered["candidate"]["image_url"],
        search_provider="demo-fixture",
        match_type="EXACT_MATCH",
    )
    h_ok = provenance_fingerprint(rec_ok)
    h_bad = provenance_fingerprint(rec_bad)
    print("=" * 70)
    print("TAMPER DEMONSTRATION - explicit fingerprint comparison")
    print("=" * 70)
    print(f"Original content fingerprint : {h_ok}")
    print(f"Tampered content fingerprint : {h_bad}")
    print(f"Fingerprints differ         : {h_ok != h_bad}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
