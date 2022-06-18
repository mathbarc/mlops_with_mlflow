import mlflow
import dotenv
import cv2
from flask import Flask, request, jsonify
import os
import base64
import io
import numpy
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = numpy.frombuffer(base64.b64decode(encoded_data), numpy.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

@app.route("/classify", methods=["POST"])
def classify():
    model = mlflow.pytorch.load_model(os.environ["MODEL_URI"])
    
    imageBase64 = request.form.get("image")
    img = readb64(imageBase64)    
    return jsonify(model.predict(img))

if __name__ == "__main__":
    dotenv.load_dotenv(".env")
    app.run(host="0.0.0.0", port="5000")
