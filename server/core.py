import tensorflow as tf
from PIL import Image
import numpy as np
from collections import Counter

KERNEL_SIZE = 224
ARCHITECTURE = 'custom'
RESCALING_FACTOR=1/255
short_symptom_labels = {
    0: 'FMD',
    1: 'IBK',
    2: 'LSD',
    3: 'NOR'
}
model = tf.keras.models.load_model(f"./{ARCHITECTURE}_model")
def resize_image(img):
    return img.resize((KERNEL_SIZE, KERNEL_SIZE), Image.ANTIALIAS)

def split_and_reshape(img):
    height, width, channels = img.shape
    img = img.reshape(1, KERNEL_SIZE,
                      1, KERNEL_SIZE, channels)
    img = img.swapaxes(1, 2)
    img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
    return img

def get_prediction(file):
    img = Image.open(file)
    img = resize_image(img)
    original_img = np.asarray(img)
    rescaled_img = original_img * RESCALING_FACTOR
    rescaled_img = split_and_reshape(rescaled_img)
    model_prediction = model.predict(rescaled_img)
    classification = np.argmax(model_prediction, axis=1)
    classification_size=len(classification)
    counter = Counter(classification)
    confidence = 0
    for i in range(classification_size):
        confidence += model_prediction[i][classification[i]]
    confidence /= classification_size
    return short_symptom_labels[counter.most_common(1)[0][0]], str(round(confidence * 100, 2))