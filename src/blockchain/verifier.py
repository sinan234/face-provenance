"""Provenance verification.

``verify_provenance`` performs the critical closing step of the pipeline:

1. re-fetch / reconstruct the discovered public content,
2. rebuild the canonical provenance record (``retrieved_at`` is recorded
   metadata and excluded from the fingerprint, so unchanged content always
   reproduces the same hash),
3. recompute the SHA-256 fingerprint,
4. query the blockchain for that fingerprint,
5. compare.

A successful check means: *this content, as retrieved, produces the exact
fingerprint that was recorded on the chain*. It proves the integrity of the
recorded fingerprint — not the truthfulness of the underlying content.
"""

from __future__ import annotations

import logging

from src.blockchain.client import BlockchainClient
from src.models.schemas import ProvenanceRecord, VerificationResult
from src.provenance.extractor import ProvenanceExtractor
from src.provenance.fingerprint import provenance_fingerprint

logger = logging.getLogger(__name__)


class ProvenanceVerifier:
    def __init__(
        self,
        client: BlockchainClient,
        extractor: ProvenanceExtractor,
    ) -> None:
        self._client = client
        self._extractor = extractor

    def verify_provenance(
        self,
        record: ProvenanceRecord,
        original_fingerprint: str | None = None,
        refetch: bool = True,
    ) -> VerificationResult:
        if refetch:
            rebuilt = self._extractor.extract(
                source_url=record.source_url,
                title_hint=record.title,
                image_url=record.image_url,
                search_provider=record.search_provider,
                match_type=record.match_type,
            )
        else:
            rebuilt = record

        calculated_hash = provenance_fingerprint(rebuilt)
        on_chain = self._client.get_record(calculated_hash)

        if on_chain is not None:
            return VerificationResult(
                verified=True,
                calculated_hash=calculated_hash,
                on_chain_hash=calculated_hash,
                transaction_hash=on_chain.transaction_hash,
                reason="Cryptographic fingerprints match",
            )

        # Content differs from what was recorded (or was never recorded).
        original = (
            self._client.get_record(original_fingerprint)
            if original_fingerprint
            else None
        )
        return VerificationResult(
            verified=False,
            calculated_hash=calculated_hash,
            on_chain_hash=original_fingerprint if original else None,
            transaction_hash=original.transaction_hash if original else None,
            reason="Content fingerprint differs from blockchain record",
        )
