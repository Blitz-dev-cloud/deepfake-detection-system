# app/predict.py - Simplified for video (image-trained model)
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from typing import Tuple, List, Optional
import tempfile

# --- Face detection setup (for images only) ---
USE_MTCNN = False
try:
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')
    USE_MTCNN = True
    print("[INFO] Using MTCNN for face detection")
except Exception:
    print("[WARN] facenet-pytorch not available, falling back to OpenCV Haar cascade")
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

from app.model import model, DEVICE

IMAGE_SIZE = 160
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def apply_clahe_pil(pil_img: Image.Image) -> Image.Image:
    """Apply CLAHE enhancement."""
    img = np.array(pil_img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img2)

def detect_and_crop_face(frame: np.ndarray) -> Optional[Image.Image]:
    """Face detection for images only."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if USE_MTCNN:
        try:
            pil = Image.fromarray(rgb)
            boxes, probs = mtcnn.detect(pil)
            if boxes is None or len(boxes) == 0:
                return None
            best_idx = np.argmax(probs)
            bbox = boxes[best_idx]
            x1, y1, x2, y2 = bbox.astype(int).tolist()
        except Exception:
            return None
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None
        areas = [w * h for (x, y, w, h) in faces]
        best_idx = np.argmax(areas)
        x, y, w, h = faces[best_idx]
        x1, y1, x2, y2 = x, y, x + w, y + h

    h_img, w_img = frame.shape[:2]
    margin = int(0.4 * max(y2 - y1, x2 - x1))
    x1m = max(0, x1 - margin)
    y1m = max(0, y1 - margin)
    x2m = min(w_img, x2 + margin)
    y2m = min(h_img, y2 + margin)
    
    crop = rgb[y1m:y2m, x1m:x2m]
    return Image.fromarray(crop)

def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    """Preprocess with CLAHE."""
    img_clahe = apply_clahe_pil(pil_img)
    return transform(img_clahe)

def tta_tensors(pil_img: Image.Image) -> List[torch.Tensor]:
    """TTA: original + flip."""
    return [preprocess_image(pil_img), preprocess_image(ImageOps.mirror(pil_img))]

def predict_image(pil_image: Image.Image, use_tta: bool = True) -> Tuple[int, float]:
    """Predict for single image."""
    model.eval()
    tensors = tta_tensors(pil_image) if use_tta else [preprocess_image(pil_image)]
    batch = torch.stack(tensors).to(DEVICE)

    with torch.no_grad():
        outputs = model(batch)
        probs = torch.softmax(outputs, dim=1)
        avg_prob = probs.mean(dim=0)
        pred_class = int(torch.argmax(avg_prob).item())
        pred_prob = float(avg_prob[pred_class].item())

    return pred_class, pred_prob

def predict_video(
    video_bytes: bytes, 
    frame_skip: int = 30,
    max_frames: int = 60,
    use_tta: bool = False  # Disabled TTA for speed
) -> Tuple[Optional[int], Optional[float], int]:
    """
    Simple video prediction - processes full frames uniformly.
    NOTE: This model was trained on images, so video results are unreliable.
    """
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(video_bytes)
    
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            print("[ERROR] Could not open video")
            return None, None, 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"[INFO] Video: {total_frames} frames, {fps:.1f} FPS")
        
        if total_frames == 0:
            cap.release()
            return None, None, 0
        
        # Sample frames uniformly
        frame_indices = list(range(0, total_frames, frame_skip))[:max_frames]
        print(f"[INFO] Sampling {len(frame_indices)} frames (every {frame_skip}th)")
        
        predictions = []
        confidences = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Process full frame (no face detection)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb)
            
            try:
                # Simple preprocessing
                tensor = transform(pil_frame).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    output = model(tensor)
                    probs = torch.softmax(output, dim=1)
                    pred_class = int(torch.argmax(probs, dim=1).item())
                    confidence = float(probs[0, pred_class].item())
                
                predictions.append(pred_class)
                confidences.append(confidence)
                
            except Exception as e:
                print(f"[WARN] Frame {frame_idx} failed: {e}")
                continue
        
        cap.release()
        
        if len(predictions) < 5:
            print(f"[ERROR] Too few predictions: {len(predictions)}")
            return None, None, 0
        
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        fake_count = np.sum(predictions == 0)
        real_count = np.sum(predictions == 1)
        
        print(f"[INFO] Results: FAKE={fake_count}, REAL={real_count}")
        print(f"[INFO] Avg confidence: {confidences.mean():.3f}")
        
        # Simple majority vote
        final_class = 0 if fake_count > real_count else 1
        final_confidence = confidences[predictions == final_class].mean()
        
        print(f"[INFO] Final: {'FAKE' if final_class == 0 else 'REAL'} ({final_confidence:.3f})")
        print(f"[WARN] Model trained on images - video results may be unreliable")
        
        return final_class, float(final_confidence), len(predictions)
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0
        
    finally:
        if 'cap' in locals():
            cap.release()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)