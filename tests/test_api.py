"""HTTP API behaviour.

The upload-guard tests do not need model weights: an oversized or
malformed upload must be rejected before it ever reaches the pipeline.
"""

from __future__ import annotations

import io

import pytest

fastapi = pytest.importorskip("fastapi")

# starlette.testclient raises at import time when no HTTP client backend is
# installed. Skip the module rather than failing collection for the suite.
try:
    from fastapi.testclient import TestClient
except (ImportError, RuntimeError) as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"fastapi TestClient unavailable: {exc}", allow_module_level=True)

from nileid.api.app import create_app  # noqa: E402
from nileid.config import Settings  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # A small cap keeps the oversize test cheap.
    app = create_app(Settings(max_upload_mb=1))
    with TestClient(app) as test_client:
        yield test_client


class TestHealth:
    def test_health_responds(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] in ("healthy", "degraded")

    def test_health_reports_version(self, client):
        assert "version" in client.get("/health").json()

    def test_openapi_schema_is_served(self, client):
        assert client.get("/openapi.json").status_code == 200


class TestUploadGuards:
    def test_rejects_a_missing_file_field(self, client):
        assert client.post("/v1/extract").status_code == 422

    def test_rejects_an_empty_file(self, client):
        response = client.post("/v1/extract", files={"file": ("empty.png", b"", "image/png")})
        assert response.status_code == 400
        assert response.json()["success"] is False

    def test_rejects_a_non_image_content_type(self, client):
        response = client.post(
            "/v1/extract",
            files={"file": ("payload.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert response.status_code == 415

    def test_rejects_an_oversized_upload(self, client):
        payload = b"\x00" * (2 * 1024 * 1024)  # 2 MB against a 1 MB cap
        response = client.post("/v1/extract", files={"file": ("big.png", payload, "image/png")})
        assert response.status_code == 413
        assert "limit" in response.json()["error"].lower()

    def test_rejects_undecodable_image_bytes(self, client):
        response = client.post(
            "/v1/extract",
            files={"file": ("broken.png", b"definitely not an image", "image/png")},
        )
        assert response.status_code in (422, 503)
        assert response.json()["success"] is False

    def test_error_bodies_are_structured(self, client):
        body = client.post("/v1/extract", files={"file": ("empty.png", b"", "image/png")}).json()
        assert set(body) == {"success", "error"}


@pytest.mark.models
class TestExtraction:
    def test_extracts_from_a_sample_card(self, client, models_available, sample_card_bytes):
        if not models_available:
            pytest.skip("YOLO weights are absent")
        response = client.post(
            "/v1/extract",
            files={"file": ("card.png", io.BytesIO(sample_card_bytes), "image/png")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert set(body["fields"]) == {
            "first_name",
            "last_name",
            "full_name",
            "address",
            "national_id",
        }
        assert "validation" in body and "warnings" in body

    def test_response_never_fabricates_derived_fields(
        self, client, models_available, sample_card_bytes
    ):
        if not models_available:
            pytest.skip("YOLO weights are absent")
        body = client.post(
            "/v1/extract",
            files={"file": ("card.png", io.BytesIO(sample_card_bytes), "image/png")},
        ).json()
        if not body["validation"]["valid"]:
            assert body["derived"]["birth_date"] is None
            assert body["derived"]["gender"] is None
