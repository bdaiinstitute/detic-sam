import io

import numpy as np
from flask import Flask, jsonify, request, send_file
import PIL
from PIL import Image


app = Flask(__name__)


def read_image(img_bytes):

    img = Image.open(io.BytesIO(img_bytes))
    img = np.array(img)
    if len(img.shape) < 3 or img.shape[2] != 3:
        raise ValueError("Not an RGB image.")
    if img.shape[0] > 2000 or img.shape[1] > 2000:
        raise ValueError("We arbitrarily limit the image size to 2000x2000 maximum.")
    return img


def np_to_bytes(arr):

    buf = io.BytesIO()
    np.savez(buf, arr1=arr, arr2=arr)
    buf.seek(0)
    return buf


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Read the received image.
        file = request.files["file"]
        img_bytes = file.read()
        try:
            image = read_image(img_bytes)
        except (PIL.UnidentifiedImageError, ValueError) as e:
            # Return an error.
            print(str(e))
            return str(e), 400

        # Get a list of classes.
        classes = request.form.get("classes")
        if classes is None:
            return "No classes specified.", 400
        classes = classes.split(",")

        image[:10, :10] = 255.
        buf = np_to_bytes(image)

        return send_file(buf, mimetype="numpy", as_attachment=True, download_name="result.npy")


if __name__ == '__main__':
    app.run(port=5550)
