import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('ff_Xception.h5')

def predict_image(model, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    if prediction[0] > 0.5:
        print("The image is a deepfake.")
    else:
        print("The image is real.")

def main():
    image_path = 'image.png'
    predict_image(model, image_path)

if __name__ == "__main__":
    main()
