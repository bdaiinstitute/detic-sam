import argparse
import json
import os
import pickle
import random
from typing import List, Dict, Tuple
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from PIL import Image
from skimage.io import imread
import torch

# Change the current working directory to 'Detic'
os.chdir('Detic')

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
sys.path.insert(0, 'third_party/CenterNet2/')

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Detic libraries 
from Detic.detic.modeling.text.text_encoder import build_text_encoder
from collections import defaultdict
from centernet.config import add_centernet_config
from Detic.detic.config import add_detic_config
from Detic.detic.modeling.utils import reset_cls_test
from sklearn.cluster import DBSCAN
import matplotlib.patches as patches
# SAM libraries
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

import io

import numpy as np
from flask import Flask, jsonify, request, send_file
import PIL
from PIL import Image

DEVICE = "cuda:0"

text_encoder = build_text_encoder(pretrain=True)
text_encoder.eval()


def DETIC_predictor():
    # Build the detector and download our pretrained weights
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3 # set threshold for this model
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
    # cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.
    detic_predictor = DefaultPredictor(cfg)

    return detic_predictor


def get_clip_embeddings(vocabulary, prompt='a '):
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb


def custom_vocab(detic_predictor, classes):
    metadata = MetadataCatalog.get("__unused2")
    metadata.thing_classes = classes # Change here to try your own vocabularies!
    classifier = get_clip_embeddings(metadata.thing_classes)
    num_classes = len(metadata.thing_classes)
    reset_cls_test(detic_predictor.model, classifier, num_classes)

    # Reset visualization threshold
    output_score_threshold = 0.3
    for cascade_stages in range(len(detic_predictor.model.roi_heads.box_predictor)):
        detic_predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold
    return metadata

def Detic(im, metadata, detic_predictor):
    if im is None:
        print("Error: Unable to read the image file")

    # Run model and show results
    output =detic_predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata)
    out = v.draw_instance_predictions(output["instances"].to('cpu'))
    instances = output["instances"].to('cpu')
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    return boxes, classes


def SAM_predictor(device):
    sam_checkpoint = "../sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = device
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor


def SAM(im, boxes, class_idx, metadata, sam_predictor):
    sam_predictor.set_image(im)
    input_boxes = torch.tensor(boxes, device=sam_predictor.device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, im.shape[:2])
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks


def read_image(img_bytes):

    img = Image.open(io.BytesIO(img_bytes))
    img = np.array(img)
    if len(img.shape) < 3 or img.shape[2] != 3:
        raise ValueError("Not an RGB image.")
    if img.shape[0] > 2000 or img.shape[1] > 2000:
        raise ValueError("We arbitrarily limit the image size to 2000x2000 maximum.")
    return img


app = Flask(__name__)

detic_predictor = DETIC_predictor()
sam_predictor = SAM_predictor(DEVICE)


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

        # Make a prediction.
        metadata = custom_vocab(detic_predictor, classes)

        boxes, class_idx = Detic(image, metadata, detic_predictor)
        if len(boxes) == 0:
            return "Did not find any objects.", 400
        masks = SAM(image, boxes, class_idx, metadata, sam_predictor)
        masks = masks.cpu().numpy()

        classes = [metadata.thing_classes[idx] for idx in class_idx]

        # Send result.
        buf = io.BytesIO()
        np.savez(buf, masks=masks, boxes=boxes, classes=classes)
        buf.seek(0)
        return send_file(buf, mimetype="numpy", as_attachment=True, download_name="result.npy")


if __name__ == '__main__':
    app.run(port=5550)
