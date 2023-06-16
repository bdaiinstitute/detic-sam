import argparse
import io
import requests
import time

import numpy as np
from PIL import Image


def image_to_bytes(img):

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def main(args):
    
    image = Image.open(args.image_path)
    buf = image_to_bytes(image)

    t1 = time.time()
    r = requests.post("http://localhost:5550/predict", files={"file": buf}, data={"classes": ",".join(args.classes)})
    print("Duration:", time.time() - t1)
    
    if r.status_code == 200:
        with io.BytesIO(r.content) as f:
            arr = np.load(f)
            print(arr["classes"])
    else:
        print(r.content)


parser = argparse.ArgumentParser()
parser.add_argument("image_path")
parser.add_argument("-c", "--classes", nargs="+")
main(parser.parse_args())
