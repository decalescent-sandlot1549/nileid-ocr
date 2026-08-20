"""OCR engine adapters and the engine ensemble."""

from nileid.ocr.engines import OCRResult, recognise, run_easyocr, run_paddleocr

__all__ = ["OCRResult", "recognise", "run_easyocr", "run_paddleocr"]
