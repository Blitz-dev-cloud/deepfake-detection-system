from PIL import Image
import io

def read_image(file) -> Image.Image:
    """Read uploaded file as PIL Image."""
    image = Image.open(io.BytesIO(file))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image
