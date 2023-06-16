import io
import requests

import numpy as np
from PIL import Image


def image_to_bytes(img):

    img = Image.fromarray(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def main():
    
    image = np.random.randint(255, size=(100, 100, 3)).astype(np.uint8)
    buf = image_to_bytes(image)
    r = requests.post("http://localhost:5550/predict", files={"file": buf}, data={"classes": "person"})
    if r.status_code == 200:
        with io.BytesIO(r.content) as f:
            arr = np.load(f)
            img = Image.fromarray(arr["arr1"])
            img.show()

main()
