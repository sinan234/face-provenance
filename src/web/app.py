"""FastAPI web interface.

Exposes the pipeline as a web API with a polished dashboard frontend:

    GET  /                 dashboard (static frontend)
    GET  /api/health       health check
    GET  /api/sample-image  bundled public-domain sample face image
    POST /api/process      run the full pipeline on an uploaded image
    POST /api/verify       re-verify (optionally with tampered demo content)

Uploads are written to a temporary directory, processed, and discarded; no
user images are stored.

Run:
    uvicorn src.web.app:app --reload --port 8000
    python -m src.web.app
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.cli import build_components
from src.models.schemas import PipelineConfig
from src.pipeline.pipeline import Pipeline, PipelineError

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
SAMPLE_IMAGE = PROJECT_ROOT / "data" / "sample_face.jpg"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

load_dotenv()

app = FastAPI(
    title="Face Provenance",
    description=(
        "Face detection + web provenance + blockchain verification. "
        "Privacy-preserving: never identifies people from faces."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "mode": os.environ.get("SEARCH_PROVIDER", "serper"),
        "search_api_key_configured": bool(os.environ.get("SEARCH_API_KEY")),
        "contract_address": os.environ.get("PROVENANCE_CONTRACT_ADDRESS") or None,
    }


@app.get("/api/sample-image")
def sample_image() -> FileResponse:
    """The bundled public-domain sample face (demo input with zero setup)."""
    if not SAMPLE_IMAGE.exists():
        raise HTTPException(status_code=404, detail="Sample image not available")
    return FileResponse(SAMPLE_IMAGE, media_type="image/jpeg")


def _run_pipeline(
    image_path: str,
    mode: str,
    chain: str,
    tamper: bool = False,
    record_on_chain: bool = True,
    image_url: str | None = None,
) -> dict:
    try:
        components = build_components(mode, chain, tamper=tamper)
        face_service, provider, validator, extractor, blockchain, config = components
        config = PipelineConfig(
            mode=mode,
            record_on_chain=record_on_chain,
            image_url=image_url or os.environ.get("SEARCH_IMAGE_URL") or None,
        )
        pipeline = Pipeline(face_service, provider, validator, extractor, blockchain, config)
        result = pipeline.run(image_path)
        return result.to_dict()
    except PipelineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc


def _save_upload(file: UploadFile) -> str:
    data = file.file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit")
    directory = Path(tempfile.mkdtemp(prefix="face-provenance-"))
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    path = directory / f"upload{suffix}"
    path.write_bytes(data)
    return str(path)


@app.post("/api/process")
async def process_image(
    file: UploadFile = File(...),
    mode: Literal["demo", "real"] = Form("demo"),
    chain: Literal["memory", "anvil"] = Form("memory"),
    image_url: str | None = Form(None),
) -> dict:
    """Run the full pipeline: face -> search -> validate -> fingerprint -> chain -> verify."""
    image_path = _save_upload(file)
    try:
        return _run_pipeline(image_path, mode, chain, image_url=image_url)
    finally:
        shutil.rmtree(Path(image_path).parent, ignore_errors=True)


@app.post("/api/verify")
async def verify_image(
    file: UploadFile = File(...),
    mode: Literal["demo", "real"] = Form("demo"),
    chain: Literal["memory", "anvil"] = Form("memory"),
    tamper: bool = Form(False),
    image_url: str | None = Form(None),
) -> dict:
    """Re-run search -> extraction -> fingerprint and verify against the chain."""
    image_path = _save_upload(file)
    try:
        return _run_pipeline(
            image_path, mode, chain, tamper=tamper, record_on_chain=False, image_url=image_url
        )
    finally:
        shutil.rmtree(Path(image_path).parent, ignore_errors=True)


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def main() -> None:
    import uvicorn

    port = int(os.environ.get("WEB_PORT", "8000"))
    uvicorn.run("src.web.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
