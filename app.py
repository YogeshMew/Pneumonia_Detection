import os
import re
import base64
from io import BytesIO

from flask import Flask, request, render_template, jsonify
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Declare a flask app
app = Flask(__name__)

MODEL_PATH = 'models/oldModel.h5'
model = load_model(MODEL_PATH)
print('Model loaded. Start serving...')


def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image


def model_predict(img, model):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='tf')
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        img.save("uploads/image.jpg")
        img_path = os.path.join(os.path.dirname(__file__), 'uploads/image.jpg')

        img = image.load_img(img_path, target_size=(64, 64))
        preds = model_predict(img, model)
        
        result = preds[0, 0]
        print(result)
        
        if result > 0.5:
            return jsonify(result="PNEUMONIA")
        else:
            return jsonify(result="NORMAL")

    return None


if __name__ == '__main__':
    app.run(port=5002, threaded=False)
