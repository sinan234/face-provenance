"""Blockchain tests.

``test_*_on_real_evm`` run against a genuine local EVM (eth-tester /
py-evm) — no network needed — deploying the actual compiled contract and
submitting real transactions. ``test_*_in_memory`` cover the explicitly
simulated offline client.
"""

from __future__ import annotations

import hashlib

import pytest
from web3 import EthereumTesterProvider, Web3

from src.blockchain.client import (
    DuplicateRecordError,
    InMemoryBlockchainClient,
    Web3BlockchainClient,
)

# Deterministic eth-tester account #0 private key (test-only, never used in
# application code and never on a real network).
ETH_TESTER_KEY_0 = "0x" + "00" * 31 + "01"


def _fingerprint(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def w3() -> Web3:
    return Web3(EthereumTesterProvider())


@pytest.fixture(scope="module")
def evm_client(w3) -> Web3BlockchainClient:
    client = Web3BlockchainClient(rpc_url="", w3=w3, chain_name="eth-tester")
    client.deploy()
    return client


# ---------------------------------------------------------------------------
# Real EVM
# ---------------------------------------------------------------------------


def test_deploy_creates_contract(evm_client) -> None:
    assert evm_client.contract_address.startswith("0x")
    assert len(evm_client.contract_address) == 42


def test_record_and_verify(evm_client) -> None:
    fingerprint = _fingerprint("record-and-verify")
    result = evm_client.record(fingerprint, source_id="https://example.com/a")
    assert result.blockchain == "eth-tester"
    assert result.fingerprint == fingerprint
    assert result.block_number >= 0
    assert result.transaction_hash.startswith("0x")
    assert result.simulated is False

    on_chain = evm_client.get_record(fingerprint)
    assert on_chain is not None
    assert on_chain.fingerprint == fingerprint
    assert on_chain.source_id == "https://example.com/a"
    assert evm_client.verify(fingerprint) is True


def test_record_submitter_is_deploying_account(evm_client, w3) -> None:
    fingerprint = _fingerprint("submitter-check")
    evm_client.record(fingerprint)
    on_chain = evm_client.get_record(fingerprint)
    assert on_chain is not None
    assert on_chain.submitter.lower() == w3.eth.accounts[0].lower()


def test_duplicate_record_rejected(evm_client) -> None:
    fingerprint = _fingerprint("duplicate")
    evm_client.record(fingerprint)
    with pytest.raises(DuplicateRecordError):
        evm_client.record(fingerprint)


def test_verify_unknown_fingerprint(evm_client) -> None:
    assert evm_client.verify(_fingerprint("never-recorded")) is False
    assert evm_client.get_record(_fingerprint("never-recorded")) is None


def test_tampered_content_not_recorded(evm_client) -> None:
    recorded = _fingerprint("original-content")
    evm_client.record(recorded)
    tampered = _fingerprint("original-content-TAMPERED")
    # Only the original fingerprint exists on chain.
    assert evm_client.verify(recorded) is True
    assert evm_client.verify(tampered) is False


def test_signed_transaction_path(w3) -> None:
    """Exercises the private-key signing path used on public testnets."""
    client = Web3BlockchainClient(
        rpc_url="", w3=w3, private_key=ETH_TESTER_KEY_0, chain_name="eth-tester"
    )
    client.deploy()
    fingerprint = _fingerprint("signed-path")
    client.record(fingerprint)
    assert client.verify(fingerprint) is True


# ---------------------------------------------------------------------------
# In-memory (explicitly simulated)
# ---------------------------------------------------------------------------


def test_in_memory_record_verify_get() -> None:
    client = InMemoryBlockchainClient()
    fingerprint = _fingerprint("memory-record")
    result = client.record(fingerprint, source_id="https://example.com/m")
    assert result.simulated is True
    assert result.blockchain == "in-memory"
    assert client.verify(fingerprint) is True
    on_chain = client.get_record(fingerprint)
    assert on_chain is not None
    assert on_chain.source_id == "https://example.com/m"
    assert client.get_record(_fingerprint("nope")) is None


def test_in_memory_duplicate_rejected() -> None:
    client = InMemoryBlockchainClient()
    fingerprint = _fingerprint("memory-duplicate")
    client.record(fingerprint)
    with pytest.raises(DuplicateRecordError):
        client.record(fingerprint)
