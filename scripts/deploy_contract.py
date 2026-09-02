"""Deploy the ProvenanceRegistry contract.

Reads chain configuration from the environment (.env):

    BLOCKCHAIN_RPC_URL          RPC endpoint (default http://127.0.0.1:8545)
    BLOCKCHAIN_PRIVATE_KEY      deploying account (Anvil dev key is fine locally)
    BLOCKCHAIN_CHAIN_ID         optional chain id override

Usage:
    python scripts/deploy_contract.py [--save-env] [--force]

``--save-env`` appends ``PROVENANCE_CONTRACT_ADDRESS=...`` to ``.env``.
``--force`` redeploys even when a contract address is already configured.

Private keys are read from the environment only — never hardcoded.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.blockchain.client import Web3BlockchainClient  # noqa: E402

logger = logging.getLogger(__name__)

ENV_FILE = PROJECT_ROOT / ".env"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-env", action="store_true",
                        help="Append PROVENANCE_CONTRACT_ADDRESS to .env")
    parser.add_argument("--force", action="store_true",
                        help="Redeploy even if a contract address is already set")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rpc_url = os.environ.get("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:8545")
    private_key = os.environ.get("BLOCKCHAIN_PRIVATE_KEY") or None
    existing = os.environ.get("PROVENANCE_CONTRACT_ADDRESS") or None

    if existing and not args.force:
        print(json.dumps({
            "already_deployed": True,
            "contract_address": existing,
            "hint": "Use --force to redeploy.",
        }, indent=2))
        return 0

    client = Web3BlockchainClient(
        rpc_url=rpc_url,
        private_key=private_key,
        chain_name="local-ethereum",
    )
    if not client.w3.is_connected():
        print(f"ERROR: cannot connect to {rpc_url}. Is Anvil running?", file=sys.stderr)
        return 1

    print(f"Deploying ProvenanceRegistry to {rpc_url} ...")
    result = client.deploy()
    print(json.dumps(result.model_dump(), indent=2))

    if args.save_env:
        _upsert_env(ENV_FILE, "PROVENANCE_CONTRACT_ADDRESS", result.contract_address)
    return 0


def _upsert_env(path: Path, key: str, value: str) -> None:
    """Set ``key=value`` in a .env file, replacing any existing line(s)."""
    line = f"{key}={value}"
    if not path.exists():
        path.write_text(line + "\n", encoding="utf-8")
        print(f"\nWrote {path}: {line}")
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = [l for l in lines if not l.strip().startswith(key + "=")]
    kept.append(line)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"\nUpdated {path}: {line}")


if __name__ == "__main__":
    sys.exit(main())
