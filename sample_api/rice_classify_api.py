# python sample_api/rice_classify_api.py

import mlflow
import dotenv
import cv2
from flask import Flask, request, jsonify
import os
import base64
import numpy
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)


def readb64(uri):
    encoded_data = uri.split(",")[1]
    nparr = numpy.frombuffer(base64.b64decode(encoded_data), numpy.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


@app.route("/classify", methods=["POST"])
def classify():
    model_uri = os.environ["MODEL_URI"]
    app.logger.info(f"Retrieving model {model_uri}")
    model = mlflow.pytorch.load_model(model_uri)
    app.logger.info("Decoding image")
    imageBase64 = request.form.get("image")
    img = readb64(imageBase64)

    app.logger.info("Inferring image")
    start = time.time()
    result = model.predict(img)
    end = time.time()

    app.logger.info(f"DONE - {end-start} s - {result}")
    return jsonify(result)


if __name__ == "__main__":
    dotenv.load_dotenv(".env")
    app.run(host="0.0.0.0", port="5000")
