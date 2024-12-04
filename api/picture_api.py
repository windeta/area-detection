import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_file
from PIL import Image
from unet_model import jaccard_coef


MODEL_PATH = 'models/satellite_unet.hdf5'
PATCH_SIZE = 256
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'jaccard_coef': jaccard_coef})

app = Flask(__name__)

def preprocess_image(image):
    image = image.resize((PATCH_SIZE, PATCH_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def postprocess_prediction(prediction):
    predicted_img = np.argmax(prediction, axis=-1)[0]
    return Image.fromarray((predicted_img * (255 // np.max(predicted_img))).astype('uint8'))


@app.route('/prediction', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    try:
        image = Image.open(file.stream).convert('RGB')
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        result_image = postprocess_prediction(prediction)

        img_io = io.BytesIO()
        result_image.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


app.run('localhost', 4000)
