"""Smart-contract artifact handling.

A precompiled artifact is committed at ``contracts/compiled/ProvenanceRegistry.json``
so no Solidity compiler is needed at runtime. If the artifact is missing (e.g.
the contract was edited), ``compile_contract()`` rebuilds it with py-solc-x,
downloading a pinned solc binary on first use.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONTRACTS_DIR = PROJECT_ROOT / "contracts"
SOL_PATH = CONTRACTS_DIR / "ProvenanceRegistry.sol"
COMPILED_ARTIFACT = CONTRACTS_DIR / "compiled" / "ProvenanceRegistry.json"
SOLC_VERSION = "0.8.24"


class ContractArtifact:
    def __init__(self, abi: list, bytecode: str, contract_name: str) -> None:
        self.abi = abi
        self.bytecode = bytecode
        self.contract_name = contract_name


def load_artifact(path: Path = COMPILED_ARTIFACT) -> ContractArtifact:
    if not path.exists():
        logger.warning("Compiled artifact missing at %s — compiling from source", path)
        return compile_contract()
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return ContractArtifact(
        abi=data["abi"],
        bytecode=data["bytecode"],
        contract_name=data["contractName"],
    )


def compile_contract(
    sol_path: Path = SOL_PATH,
    solc_version: str = SOLC_VERSION,
) -> ContractArtifact:
    """Compile ProvenanceRegistry.sol with py-solc-x."""
    import solcx  # imported lazily: only needed for compilation

    try:
        solcx.get_installed_solc_versions()
    except Exception:
        solcx.install_solc(solc_version)
    if solc_version not in {str(v) for v in solcx.get_installed_solc_versions()}:
        solcx.install_solc(solc_version)
    solcx.set_solc_version(solc_version)

    compiled = solcx.compile_files(
        [str(sol_path)],
        output_values=["abi", "bin"],
        solc_version=solc_version,
    )
    for key, contract in compiled.items():
        if key.endswith("ProvenanceRegistry"):
            return ContractArtifact(
                abi=contract["abi"],
                bytecode=contract["bin"],
                contract_name="ProvenanceRegistry",
            )
    raise RuntimeError("Compilation did not produce a ProvenanceRegistry contract")


def save_artifact(artifact: ContractArtifact, path: Path = COMPILED_ARTIFACT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "contractName": artifact.contract_name,
        "abi": artifact.abi,
        "bytecode": artifact.bytecode,
        "compiler": f"solc {SOLC_VERSION}",
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path
