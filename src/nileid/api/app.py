"""FastAPI service exposing the reader over HTTP.

Security and privacy posture:

* Uploads are held in memory and never written to disk. Nothing about a
  submitted card is persisted by this service.
* Only the size and MIME type of an upload are logged -- never filenames
  (which frequently contain a person's name) and never extracted values.
* Uploads are capped at ``Settings.max_upload_mb`` and the body is read in
  bounded chunks, so an oversized upload is rejected before it is buffered.
* Models are loaded once during start-up rather than on the first request.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from nileid import __version__
from nileid.config import DEFAULT_SETTINGS, Settings
from nileid.logging_utils import configure_logging, get_logger
from nileid.models import ModelNotFoundError, missing_models
from nileid.pipeline.reader import EgyptianIDReader
from nileid.preprocessing.image_io import ImageLoadError

log = get_logger("api")

#: Chunk size used when streaming an upload into memory.
_CHUNK = 64 * 1024

_state: dict[str, object] = {}


def get_reader() -> EgyptianIDReader:
    """Return the process-wide reader, creating it on first use."""
    reader = _state.get("reader")
    if reader is None:
        reader = EgyptianIDReader(_state.get("settings") or DEFAULT_SETTINGS)
        _state["reader"] = reader
    return reader  # type: ignore[return-value]


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001 - required by FastAPI
    settings: Settings = _state.get("settings") or DEFAULT_SETTINGS  # type: ignore[assignment]
    configure_logging(settings.log_level)
    absent = missing_models(settings)
    if absent:
        log.error(
            "Missing model weights: %s. Run: python scripts/download_models.py",
            ", ".join(absent),
        )
    else:
        # Pay the load cost before the first request rather than during it.
        try:
            from nileid.models import warmup

            warmup(settings)
            log.info("Models loaded; service ready.")
        except Exception:
            log.exception("Warm-up failed; models will load on first request.")
    yield
    _state.clear()


router = APIRouter()


@router.get("/health", tags=["service"])
async def health() -> dict:
    """Liveness probe."""
    settings: Settings = _state.get("settings") or DEFAULT_SETTINGS  # type: ignore[assignment]
    absent = missing_models(settings)
    return {
        "status": "degraded" if absent else "healthy",
        "version": __version__,
        "missing_models": absent,
    }


@router.post("/v1/extract", tags=["extraction"])
async def extract(file: UploadFile = File(..., description="Card image")) -> JSONResponse:
    """Extract structured data from an Egyptian National ID card image.

    Always returns a structured body. A card that cannot be read yields
    ``card_detected: false`` with warnings explaining why, rather than an
    opaque error.
    """
    settings: Settings = _state.get("settings") or DEFAULT_SETTINGS  # type: ignore[assignment]

    content_type = (file.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        return _error(f"Unsupported content type {content_type!r}; expected an image.", 415)

    # Read with a hard ceiling so a large upload cannot exhaust memory.
    data = bytearray()
    while True:
        chunk = await file.read(_CHUNK)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > settings.max_upload_bytes:
            log.warning("Rejected an upload over the %d MB limit.", settings.max_upload_mb)
            return _error(f"File exceeds the {settings.max_upload_mb} MB limit.", 413)

    if not data:
        return _error("The uploaded file is empty.", 400)

    log.info("Processing an upload: %d bytes, content_type=%s", len(data), content_type or "-")

    try:
        result = get_reader().read(bytes(data))
    except ImageLoadError as exc:
        return _error(str(exc), 422)
    except ModelNotFoundError as exc:
        log.error("%s", exc)
        return _error("The service is not configured with model weights.", 503)
    except Exception:
        # Deliberately generic: an internal message could echo image content.
        log.exception("Unhandled error while processing an upload.")
        return _error("An internal error occurred while processing the image.", 500)

    return JSONResponse(status_code=200, content={"success": True, **result.to_dict()})


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse(status_code=status, content={"success": False, "error": message})


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory."""
    _state["settings"] = settings or DEFAULT_SETTINGS
    application = FastAPI(
        title="NileID",
        version=__version__,
        description=(
            "Structured information extraction from Egyptian National ID cards.\n\n"
            "Uploaded images are processed in memory and are never stored."
        ),
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


app = create_app()
