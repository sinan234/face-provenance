// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title ProvenanceRegistry
/// @notice Stores the minimum cryptographic proof necessary to attest that a
///         piece of web content existed at a point in time.
///
///         The contract deliberately stores ONLY:
///           - the SHA-256 provenance fingerprint (bytes32),
///           - the submitting account,
///           - the submission timestamp / block,
///           - an optional public source identifier (e.g. the source URL).
///
///         It NEVER stores face embeddings, personal information, private
///         images, or private social-media data. Recording a fingerprint is
///         an integrity proof, not a statement about content truthfulness.
contract ProvenanceRegistry {
    struct Record {
        bytes32 fingerprint;
        address submitter;
        uint256 timestamp;
        uint256 blockNumber;
        string sourceId;
        bool exists;
    }

    mapping(bytes32 => Record) private _records;
    uint256 public recordCount;

    event ProvenanceRecorded(
        bytes32 indexed fingerprint,
        address indexed submitter,
        uint256 timestamp,
        uint256 blockNumber,
        string sourceId
    );

    /// @notice Records a provenance fingerprint. Reverts if already recorded.
    /// @param fingerprint SHA-256 provenance fingerprint (bytes32).
    /// @param sourceId Optional public source identifier (e.g. source URL).
    function record(bytes32 fingerprint, string calldata sourceId)
        external
        returns (bool)
    {
        require(fingerprint != bytes32(0), "ProvenanceRegistry: empty fingerprint");
        require(!_records[fingerprint].exists, "ProvenanceRegistry: already recorded");

        _records[fingerprint] = Record({
            fingerprint: fingerprint,
            submitter: msg.sender,
            timestamp: block.timestamp,
            blockNumber: block.number,
            sourceId: sourceId,
            exists: true
        });
        recordCount += 1;

        emit ProvenanceRecorded(
            fingerprint,
            msg.sender,
            block.timestamp,
            block.number,
            sourceId
        );
        return true;
    }

    /// @notice True if the fingerprint has been recorded.
    function verify(bytes32 fingerprint) external view returns (bool) {
        return _records[fingerprint].exists;
    }

    /// @notice Returns the full record for a fingerprint. Reverts if absent.
    function getRecord(bytes32 fingerprint)
        external
        view
        returns (
            address submitter,
            uint256 timestamp,
            uint256 blockNumber,
            string memory sourceId
        )
    {
        require(_records[fingerprint].exists, "ProvenanceRegistry: not found");
        Record storage r = _records[fingerprint];
        return (r.submitter, r.timestamp, r.blockNumber, r.sourceId);
    }
}
