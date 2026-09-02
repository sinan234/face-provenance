"""Blockchain access layer.

Two implementations of the ``BlockchainClient`` protocol:

- ``Web3BlockchainClient`` — talks to any Ethereum-compatible chain
  (Anvil locally). Deploys the ``ProvenanceRegistry``
  contract and submits/reads fingerprints.
- ``InMemoryBlockchainClient`` — an explicitly-labelled simulation for tests
  and the offline demo. Every result it produces carries
  ``simulated=True`` and its transaction hashes are deterministic
  placeholders, so it can never be mistaken for a real chain.

Private keys are never hardcoded: they come from environment configuration.
The signing account is only used when a private key is supplied.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Protocol

from src.blockchain.contract import ContractArtifact, load_artifact
from src.models.schemas import BlockchainRecordResult, OnChainRecord

logger = logging.getLogger(__name__)


class BlockchainError(Exception):
    """Base error for blockchain operations."""


class DuplicateRecordError(BlockchainError):
    """The fingerprint is already recorded on the chain."""


class ContractNotDeployedError(BlockchainError):
    """No contract address configured/known."""


class BlockchainClient(Protocol):
    blockchain_name: str

    def deploy(self) -> BlockchainRecordResult:
        ...

    def record(self, fingerprint: str, source_id: str | None = None) -> BlockchainRecordResult:
        ...

    def get_record(self, fingerprint: str) -> OnChainRecord | None:
        ...

    def verify(self, fingerprint: str) -> bool:
        ...


# ---------------------------------------------------------------------------
# Web3 client (real chains)
# ---------------------------------------------------------------------------


class Web3BlockchainClient:
    """Ethereum-compatible chain client (Anvil / Ganache)."""

    blockchain_name = "ethereum"

    def __init__(
        self,
        rpc_url: str,
        private_key: str | None = None,
        contract_address: str | None = None,
        artifact: ContractArtifact | None = None,
        chain_name: str = "ethereum",
        w3=None,
    ) -> None:
        if w3 is None:
            from web3 import HTTPProvider, Web3

            w3 = Web3(HTTPProvider(rpc_url))
        self.w3 = w3
        self._artifact = artifact or load_artifact()
        self._chain_name = chain_name

        self._account = None
        if private_key:
            self._account = self.w3.eth.account.from_key(private_key)
            logger.info("Blockchain client using account %s", self._account.address)

        if contract_address:
            self.contract_address = contract_address
        else:
            self.contract_address = None

    # -- deployment ---------------------------------------------------------
    def deploy(self) -> BlockchainRecordResult:
        if not self.w3.is_connected():
            raise BlockchainError(
                f"Cannot reach the chain at {getattr(self.w3.provider, 'endpoint_uri', 'unknown RPC')}. "
                "Start Anvil (`anvil` or `docker compose up -d anvil`) and check "
                "BLOCKCHAIN_RPC_URL."
            )
        if not self._artifact.bytecode:
            raise BlockchainError("Artifact has no bytecode; cannot deploy")
        contract = self.w3.eth.contract(
            abi=self._artifact.abi, bytecode=self._artifact.bytecode
        )
        tx_hash = self._submit(contract.constructor())
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt.status == 0:
            raise BlockchainError("Contract deployment transaction reverted")
        address = receipt.contractAddress
        if not address:
            raise BlockchainError("Deployment produced no contract address")
        self.contract_address = address
        self._persist_contract_address(address)
        logger.info("Deployed ProvenanceRegistry at %s", address)
        return BlockchainRecordResult(
            blockchain=self._chain_name,
            contract_address=address,
            transaction_hash=_to_0x_hex(tx_hash),
            block_number=receipt.blockNumber,
            fingerprint="",
        )

    # -- records ------------------------------------------------------------
    def record(self, fingerprint: str, source_id: str | None = None) -> BlockchainRecordResult:
        self._ensure_contract()
        contract = self._contract()
        fp_bytes = self._fingerprint_bytes32(fingerprint)
        source_id = source_id or ""
        try:
            tx_hash = self._submit(contract.functions.record(fp_bytes, source_id))
        except Exception as exc:
            if "already recorded" in str(exc):
                raise DuplicateRecordError(
                    f"Fingerprint already recorded: {fingerprint}"
                ) from exc
            raise BlockchainError(f"record() failed: {exc}") from exc
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt.status == 0:
            reason = self._revert_reason(
                contract.functions.record, (fp_bytes, source_id)
            )
            if "already recorded" in reason:
                raise DuplicateRecordError(
                    f"Fingerprint already recorded: {fingerprint}"
                )
            raise BlockchainError(
                f"record() transaction reverted: {reason}"
            )
        return BlockchainRecordResult(
            blockchain=self._chain_name,
            contract_address=self.contract_address,
            transaction_hash=_to_0x_hex(tx_hash),
            block_number=receipt.blockNumber,
            fingerprint=fingerprint,
        )

    def get_record(self, fingerprint: str) -> OnChainRecord | None:
        contract = self._contract()
        fp_bytes = self._fingerprint_bytes32(fingerprint)
        try:
            submitter, timestamp, block_number, source_id = contract.functions.getRecord(
                fp_bytes
            ).call()
        except Exception as exc:
            if "not found" in str(exc):
                return None
            logger.debug("getRecord failed: %s", exc)
            return None
        return OnChainRecord(
            fingerprint=fingerprint,
            submitter=submitter,
            timestamp=timestamp,
            block_number=block_number,
            source_id=source_id,
            transaction_hash=self._find_transaction_hash(fingerprint),
        )

    def verify(self, fingerprint: str) -> bool:
        contract = self._contract()
        return bool(contract.functions.verify(self._fingerprint_bytes32(fingerprint)).call())

    # -- internals ----------------------------------------------------------
    def _is_local_chain(self) -> bool:
        """Local EVMs (Anvil/Hardhat, eth-tester) are safe to auto-deploy on."""
        return self._chain_name in ("local-anvil", "local-ethereum", "eth-tester")

    def _has_code(self, address: str) -> bool:
        try:
            return bool(self.w3.eth.get_code(address))
        except Exception as exc:
            logger.debug("eth_getCode failed for %s: %s", address, exc)
            return False

    def _ensure_contract(self) -> None:
        """Make sure a usable contract exists before recording.

        If the configured address holds no code (e.g. the local node was
        restarted and the recorded address belongs to a previous chain), a
        local chain gets a fresh deployment automatically; other chains raise
        a clear error instead of silently writing nothing.
        """
        if self.contract_address and self._has_code(self.contract_address):
            return
        if self._is_local_chain():
            logger.warning(
                "No ProvenanceRegistry code at %s on %s - deploying now",
                self.contract_address or "(none)",
                self._chain_name,
            )
            self.deploy()
            return
        if self.contract_address:
            raise ContractNotDeployedError(
                "No contract code found at configured address "
                f"{self.contract_address}. Deploy with "
                "scripts/deploy_contract.py --save-env."
            )
        raise ContractNotDeployedError(
            "No contract address configured. Run scripts/deploy_contract.py "
            "or set PROVENANCE_CONTRACT_ADDRESS."
        )

    def _revert_reason(self, fn, args: tuple) -> str:
        """Best-effort revert message for a mined-but-reverted transaction."""
        try:
            fn(*args).call({"from": self._default_account()})
            return "unknown revert"
        except Exception as exc:
            return str(exc)

    def _persist_contract_address(self, address: str) -> None:
        """Keep .env in sync so later processes reuse the deployed contract."""
        if not self._is_local_chain():
            return
        try:
            env_file = Path(__file__).resolve().parents[2] / ".env"
            if not env_file.exists():
                return
            lines = env_file.read_text(encoding="utf-8").splitlines()
            kept = [l for l in lines if not l.strip().startswith("PROVENANCE_CONTRACT_ADDRESS=")]
            kept.append(f"PROVENANCE_CONTRACT_ADDRESS={address}")
            env_file.write_text("\n".join(kept) + "\n", encoding="utf-8")
        except Exception as exc:
            logger.debug("Could not persist contract address to .env: %s", exc)

    def _contract(self):
        if not self.contract_address:
            raise ContractNotDeployedError(
                "No contract address configured. Run scripts/deploy_contract.py "
                "or set PROVENANCE_CONTRACT_ADDRESS."
            )
        return self.w3.eth.contract(
            address=self.contract_address, abi=self._artifact.abi
        )

    def _default_account(self) -> str:
        if self._account:
            return self._account.address
        if self.w3.eth.accounts:
            return self.w3.eth.accounts[0]
        raise BlockchainError("No account available to submit transactions")

    def _submit(self, fn):
        """Submit a contract function call; sign when a private key exists.

        ``fn`` is a web3 ``ContractFunction`` (or ``ContractConstructor`` for
        deployment) exposing ``transact`` and ``build_transaction``.
        """
        if self._account is None:
            # Unlocked local account (Anvil dev accounts, eth-tester).
            return fn.transact({"from": self._default_account()})
        # Signed path (remote nodes with a funded key).
        address = self._account.address
        transaction = fn.build_transaction(
            {
                "from": address,
                "nonce": self.w3.eth.get_transaction_count(address),
                "chainId": self.w3.eth.chain_id,
            }
        )
        signed = self._account.sign_transaction(transaction)
        raw = getattr(signed, "raw_transaction", None) or getattr(
            signed, "rawTransaction", None
        )
        return self.w3.eth.send_raw_transaction(raw)

    def _find_transaction_hash(self, fingerprint: str) -> str | None:
        """Look up the recording transaction via the ProvenanceRecorded event."""
        try:
            event = self._contract().events.ProvenanceRecorded
            logs = event.get_logs(
                from_block=0,
                argument_filters={"fingerprint": self._fingerprint_bytes32(fingerprint)},
            )
            if logs:
                return _to_0x_hex(logs[0]["transactionHash"])
        except Exception as exc:
            logger.debug("Event lookup failed: %s", exc)
        return None

    @staticmethod
    def _fingerprint_bytes32(fingerprint: str) -> bytes:
        try:
            return bytes.fromhex(fingerprint)
        except ValueError as exc:
            raise BlockchainError(f"Invalid fingerprint hex: {fingerprint}") from exc


def _to_0x_hex(value) -> str:
    """Normalize HexBytes/bytes to a 0x-prefixed hex string."""
    text = value.hex() if hasattr(value, "hex") else str(value)
    return text if text.startswith("0x") else "0x" + text


# ---------------------------------------------------------------------------
# In-memory client (explicitly simulated)
# ---------------------------------------------------------------------------

_MEMORY_ADDRESS = "0x" + "00" * 20
_MEMORY_SUBMITTER = "0x" + "11" * 20


class InMemoryBlockchainClient:
    """Offline, explicitly-labelled simulation of the registry chain."""

    blockchain_name = "in-memory"

    def __init__(self, contract_address: str = _MEMORY_ADDRESS) -> None:
        self._records: dict[str, OnChainRecord] = {}
        self._counter = 0
        self.contract_address = contract_address

    def deploy(self) -> BlockchainRecordResult:
        self.contract_address = _MEMORY_ADDRESS
        return BlockchainRecordResult(
            blockchain=self.blockchain_name,
            contract_address=self.contract_address,
            transaction_hash="0x" + "0" * 64,
            block_number=0,
            fingerprint="",
            simulated=True,
        )

    def record(self, fingerprint: str, source_id: str | None = None) -> BlockchainRecordResult:
        if fingerprint in self._records:
            raise DuplicateRecordError(f"Fingerprint already recorded: {fingerprint}")
        self._counter += 1
        # Deterministic placeholder hash — clearly not a real chain tx.
        tx_hash = "0x" + hashlib.sha256(
            f"{fingerprint}:{self._counter}:in-memory".encode("utf-8")
        ).hexdigest()
        record = OnChainRecord(
            fingerprint=fingerprint,
            submitter=_MEMORY_SUBMITTER,
            timestamp=int(time.time()),
            block_number=self._counter,
            source_id=source_id,
            transaction_hash=tx_hash,
        )
        self._records[fingerprint] = record
        return BlockchainRecordResult(
            blockchain=self.blockchain_name,
            contract_address=self.contract_address,
            transaction_hash=tx_hash,
            block_number=self._counter,
            fingerprint=fingerprint,
            simulated=True,
        )

    def get_record(self, fingerprint: str) -> OnChainRecord | None:
        return self._records.get(fingerprint)

    def verify(self, fingerprint: str) -> bool:
        return fingerprint in self._records
