from PIL import Image
import io

def read_image(file_bytes: bytes) -> Image.Image:
    """
    Convert uploaded file bytes to PIL Image
    """
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")
