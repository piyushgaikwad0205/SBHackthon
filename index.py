import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('celebs_Xception.h5')

def predict_image(model, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    prediction_value = prediction[0][0] 
    if prediction_value > 0.5:
        print("The image is a deepfake.")
    else:
        print("The image is real.")

def main():
    image_path = 'Screenshot 2024-10-05 125418.png'
    predict_image(model, image_path)

if __name__ == "__main__":
    main()
