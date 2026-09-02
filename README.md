# Face Provenance

Privacy-preserving **face detection → genuine reverse-image search → blockchain-verified provenance** pipeline.

Takes a face image, detects the face (never identifies who it is), reverse-searches the image against real public web content, builds a canonical SHA-256 fingerprint of the matching post, records it on an Ethereum-compatible blockchain, then re-fetches, recomputes, and verifies the fingerprint — **PASS / FAIL**.

## Pipeline

```
input image → face detection → reverse-image search (Serper)
  → candidate validation → canonical fingerprint (SHA-256)
  → blockchain record → re-fetch → recompute → VERIFIED / FAILED
```

## Privacy

This system **never identifies, names, profiles, or tracks people from faces**. It only reports face presence, location, and a technical similarity score. Only the cryptographic fingerprint (never the image, face, or personal data) is stored on-chain.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: python -m venv .venv / .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                  # then add your SEARCH_API_KEY (serper.dev)
```

### Real pipeline (recommended)

```bash
npx hardhat node                      # terminal 1: start the local chain (Anvil) on :8545
python scripts/deploy_contract.py --save-env   # deploy ProvenanceRegistry

python -m src.cli process data/sample_face.jpg --mode real --chain anvil
```

### Web dashboard

```bash
python -m src.web.app                 # open http://localhost:8000
```

### Tests

```bash
pytest
```

## Blockchain

The pipeline runs against an **Ethereum-compatible chain**:

| Chain | What it is | When to use |
|---|---|---|
| `memory` | in-RAM simulation (labelled SIMULATED) | offline tests |
| **`anvil`** | **real local Ethereum (Hardhat node)** — recommended | demos, no internet/funds needed |
| `testnet` | public network (e.g. Sepolia) | optional; needs RPC + funded private key |

`contracts/ProvenanceRegistry.sol` stores only `(fingerprint, submitter, timestamp, block, sourceId)` — the minimum proof needed.

## Limitations

- Reverse-image search needs a Serper API key and finds only content that is publicly indexed.
- Some sites block automated access or serve dynamic HTML, so a match may report `NO MATCH` rather than fabricate one.
- Public pages can change or disappear — re-verification compares against the *recorded* fingerprint.
- Perceptual similarity is not proof of identity.
- The blockchain stores a fingerprint, not the original content; confirmation proves the fingerprint was recorded, not that the content is truthful.

## License

MIT
