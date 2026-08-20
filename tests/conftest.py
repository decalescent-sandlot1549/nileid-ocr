"""Shared fixtures.

Every image used by the tests is generated at run time by
``scripts/make_sample_card.py``. No identity document, real or
photographed, is stored in this repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from nileid.config import Settings  # noqa: E402
from nileid.models import missing_models  # noqa: E402

#: A structurally valid, entirely synthetic National ID number.
#: 1990-01-01, Cairo (01), odd sequence digit -> male.
SAMPLE_NID = "29001010100017"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def models_available(settings: Settings) -> bool:
    return not missing_models(settings)


@pytest.fixture(scope="session")
def sample_card_bgr() -> np.ndarray:
    """A synthetic card face as an OpenCV BGR array."""
    import cv2
    from make_sample_card import render_card

    rgb = np.asarray(render_card(national_id=SAMPLE_NID))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@pytest.fixture(scope="session")
def sample_card_png(tmp_path_factory) -> Path:
    """The synthetic card written to a temporary PNG."""
    from make_sample_card import render_card

    path = tmp_path_factory.mktemp("cards") / "sample_card.png"
    render_card(national_id=SAMPLE_NID).save(path)
    return path


@pytest.fixture(scope="session")
def sample_card_bytes(sample_card_png: Path) -> bytes:
    return sample_card_png.read_bytes()


@pytest.fixture
def blank_image() -> np.ndarray:
    """A featureless image containing no card."""
    return np.full((480, 640, 3), 200, dtype=np.uint8)


requires_models = pytest.mark.skipif(
    bool(missing_models(Settings())),
    reason="YOLO weights are absent; run scripts/download_models.py",
)
