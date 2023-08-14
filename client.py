import argparse
import io
import requests
import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def ask_sam(image, classes):
    buf = image_to_bytes(image)
    r = requests.post("http://localhost:5550/predict",
                      files={"file": buf},
                      data={"classes": ",".join(classes)}
                     )

    if r.status_code != 200:
        assert False, r.content

    with io.BytesIO(r.content) as f:
        arr = np.load(f, allow_pickle=True)
        boxes = arr['boxes']
        classes = arr['classes']
        masks = arr['masks']
        scores = arr['scores']

    return dict(boxes=boxes, classes=classes, masks=masks, scores=scores)


def image_to_bytes(img):

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def visualize_output(im, masks, input_boxes, classes, scores):
    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box, class_name, score in zip(input_boxes, classes, scores):
        show_box(box, plt.gca())
        x, y = box[:2]
        plt.gca().text(x, y - 5, class_name + f': {score:.2f}', color='white', fontsize=12, fontweight='bold', bbox=dict(facecolor='green', edgecolor='green', alpha=0.5))
    plt.axis('off')
    plt.show()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def main(args):
    
    image = Image.open(args.image_path)
    d = ask_sam(image, args.classes)
    visualize_output(image, d["masks"], d["boxes"], d["classes"], d['scores'])


parser = argparse.ArgumentParser()
parser.add_argument("image_path")
parser.add_argument("-c", "--classes", nargs="+")
main(parser.parse_args())
