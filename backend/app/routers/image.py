import io

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from ..database import save_check
from ..services.image_detector import get_image_detector

router = APIRouter(tags=["image"])

MAX_IMAGE_BYTES = 15 * 1024 * 1024


def read_image(data: bytes, name: str = "image") -> Image.Image:
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"{name} is larger than 15 MB.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, f"{name} is not a valid image file.")


@router.post("/image/analyze")
def analyze_image(file: UploadFile = File(...), explain: bool = Form(True)):
    # Plain `def` so FastAPI runs the heavy model work in a worker thread.
    image = read_image(file.file.read(), file.filename or "image")
    detector = get_image_detector()
    result = detector.predict_with_gradcam(image) if explain else detector.predict(image)
    result["filename"] = file.filename
    result["id"] = save_check(
        kind="image",
        input_preview=file.filename or "image",
        verdict=result["label"],
        confidence=result["confidence"],
        details={"probabilities": result["probabilities"]},
    )
    return result
