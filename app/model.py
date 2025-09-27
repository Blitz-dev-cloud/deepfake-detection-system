import tensorflow as tf
import numpy as np

# Load saved model once at startup
MODEL_PATH = "models/deepfake_detector_mobilenet_tfdata.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (112, 112)

def predict_image(image):
    """
    Takes a PIL image, preprocesses it, and returns prediction + probability.
    """
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # (1, 112, 112, 3)

    pred_prob = model.predict(image)[0][0]
    prediction = 1 if pred_prob > 0.5 else 0
    return prediction, float(pred_prob)
