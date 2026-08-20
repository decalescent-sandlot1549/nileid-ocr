"""Streamlit demo for NileID.

Run with:
    streamlit run demo/app.py

The demo processes uploads in memory and writes nothing to disk.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import streamlit as st

# Allow running straight from a checkout without installing the package.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nileid import EgyptianIDReader, Settings  # noqa: E402
from nileid.models import missing_models  # noqa: E402
from nileid.preprocessing.image_io import ImageLoadError  # noqa: E402
from nileid.results import Status  # noqa: E402

st.set_page_config(page_title="NileID", page_icon="🪪", layout="wide")

STATUS_LABEL = {
    Status.OK: ("High confidence", "#1a7f4b"),
    Status.LOW_CONFIDENCE: ("Low confidence", "#a06a00"),
    Status.UNREADABLE: ("Unreadable", "#9c2b2b"),
    Status.NOT_FOUND: ("Not found", "#6b7280"),
}

CSS = """
<style>
  .block-container { padding-top: 2.5rem; max-width: 1200px; }
  .nid-field {
      border: 1px solid rgba(128,128,128,.25); border-radius: 8px;
      padding: .7rem .9rem; margin-bottom: .55rem;
  }
  .nid-label { font-size: .74rem; text-transform: uppercase;
      letter-spacing: .06em; opacity: .65; margin-bottom: .25rem; }
  .nid-value { font-size: 1.12rem; font-weight: 600; }
  .nid-value.rtl { direction: rtl; text-align: right; font-family:
      "Segoe UI", "Tahoma", sans-serif; }
  .nid-value.mono { font-family: ui-monospace, "Cascadia Code", monospace;
      letter-spacing: .09em; }
  .nid-value.empty { opacity: .4; font-weight: 400; font-style: italic; }
  .nid-badge { float: right; font-size: .68rem; padding: .1rem .45rem;
      border-radius: 10px; color: #fff; font-weight: 600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_reader(device: str, fuzzy: bool) -> EgyptianIDReader:
    """Build the reader once and reuse it across reruns."""
    return EgyptianIDReader(Settings(device=device, enable_fuzzy_correction=fuzzy))


def field_card(label: str, field, *, rtl: bool = False, mono: bool = False) -> None:
    text, colour = STATUS_LABEL[field.status]
    badge = (
        f'<span class="nid-badge" style="background:{colour}">'
        f"{text} · {field.confidence:.0%}</span>"
        if field.value
        else f'<span class="nid-badge" style="background:{colour}">{text}</span>'
    )
    classes = "nid-value" + (" rtl" if rtl else "") + (" mono" if mono else "")
    if field.value:
        value = f'<div class="{classes}">{field.value}</div>'
    else:
        value = '<div class="nid-value empty">not read</div>'
    st.markdown(
        f'<div class="nid-field"><div class="nid-label">{label}{badge}</div>{value}</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("NileID")
    st.caption(
        "Structured information extraction from Egyptian National ID cards. "
        "Images are processed in memory and are not stored."
    )

    with st.sidebar:
        st.subheader("Settings")
        device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
        fuzzy = st.checkbox(
            "Dictionary-assisted Arabic correction",
            value=False,
            help=(
                "Snaps tokens to a lexicon of common names and places. Improves "
                "frequent names and can corrupt rare ones, so it is off by default."
            ),
        )
        st.divider()
        st.caption(
            "Upload a **synthetic or redacted** card. Do not upload a real "
            "identity document to a demo you do not control."
        )

    absent = missing_models(Settings())
    if absent:
        st.error(
            "Model weights are missing: "
            + ", ".join(absent)
            + "\n\nRun `python scripts/download_models.py` and restart."
        )
        return

    uploaded = st.file_uploader(
        "Card image", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"]
    )
    sample = Path(__file__).resolve().parents[1] / "examples" / "sample_card.png"
    use_sample = st.button("Use the bundled sample card", disabled=not sample.exists())

    data: bytes | None = None
    if uploaded is not None:
        data = uploaded.getvalue()
    elif use_sample and sample.exists():
        data = sample.read_bytes()

    if data is None:
        st.info("Upload a card image, or load the bundled synthetic sample.")
        return

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Input")
        st.image(io.BytesIO(data), width="stretch")

    with right:
        st.subheader("Extracted fields")
        reader = get_reader(device, fuzzy)
        with st.spinner("Processing…"):
            try:
                result = reader.read(data)
            except ImageLoadError as exc:
                st.error(f"Could not read that image: {exc}")
                return

        if not result.card_detected:
            st.warning(
                "No card outline was detected. The image was processed as if it "
                "were already cropped to the card."
            )

        field_card("Full name", result.full_name, rtl=True)
        field_card("Address", result.address, rtl=True)
        field_card("National ID", result.national_id, mono=True)

        if result.birth_date:
            a, b, c = st.columns(3)
            a.metric("Birth date", result.birth_date)
            b.metric("Governorate", result.governorate or "—")
            c.metric("Gender", (result.gender or "—").title())
        else:
            st.info(
                "Birth date, governorate and gender are derived from the National "
                "ID number and are shown only when it passes validation."
            )

        validation = result.validation
        if validation.valid:
            st.success("National ID passes structural validation.")
        elif result.national_id.value:
            st.error("National ID failed validation: " + "; ".join(validation.errors))

        if result.warnings:
            with st.expander(f"Warnings ({len(result.warnings)})", expanded=True):
                for warning in result.warnings:
                    st.write("·", warning)

        with st.expander("Raw JSON"):
            st.code(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), language="json")

        st.download_button(
            "Download JSON",
            data=json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            file_name="nileid_result.json",
            mime="application/json",
        )
        st.caption(f"Processed in {result.processing_time_ms:.0f} ms")


main()
