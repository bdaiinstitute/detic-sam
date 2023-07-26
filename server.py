import io
import os
import sys

import numpy as np
from PIL import Image
import torch
import numpy as np
from flask import Flask, request, send_file
import PIL
from PIL import Image

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
from centernet.config import add_centernet_config
from Detic.detic.config import add_detic_config
from Detic.detic.modeling.utils import reset_cls_test

# SAM libraries
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

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
    classifier = get_clip_embeddings(classes)
    num_classes = len(classes)
    reset_cls_test(detic_predictor.model, classifier, num_classes)

    # Reset visualization threshold
    output_score_threshold = 0.3
    for cascade_stages in range(len(detic_predictor.model.roi_heads.box_predictor)):
        detic_predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold


def Detic(im, detic_predictor):
    if im is None:
        print("Error: Unable to read the image file")

    # Run model and show results
    output =detic_predictor(im[:, :, ::-1])  # Detic expects BGR images.
    # v = Visualizer(im[:, :, ::-1], metadata)
    # out = v.draw_instance_predictions(output["instances"].to('cpu'))
    instances = output["instances"].to('cpu')
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    return boxes, classes, scores


def SAM_predictor(device):
    sam_checkpoint = "../sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = device
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor


def SAM(im, boxes, sam_predictor):
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
        custom_vocab(detic_predictor, classes)

        boxes, class_idx, scores = Detic(image, detic_predictor)
        if len(boxes) == 0:
            return "Did not find any objects.", 400
        masks = SAM(image, boxes, sam_predictor)
        masks = masks.cpu().numpy()

        classes = [classes[idx] for idx in class_idx]

        # Send result.
        buf = io.BytesIO()
        np.savez(buf, masks=masks, boxes=boxes, classes=classes, scores=scores)
        buf.seek(0)
        return send_file(buf, mimetype="numpy", as_attachment=True, download_name="result.npy")


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """Given a batch of images, runs the detector on all of these
    and returns the result.
    
    This method expects the request to be in the form of a dictionary
    whose keys are strings corresponding to the camera names, and whose
    values are numpy arrays corresponding to the RGB image taken from this
    camera.
    
    This method primarily exists to support speedy inference for the SeSaMe
    planning system (https://github.com/bdaiinstitute/predicators), but more
    generally hsould supposrt a perception pipeline that's part of some
    broader planning system."""

    if request.method == "POST":
        # Get a list of classes.
        classes = request.form.get("classes")
        if classes is None:
            return "No classes specified.", 400
        classes = classes.split(",")
        # Make a prediction.
        # TODO: consider caching this as long as the classes
        # are the same!
        custom_vocab(detic_predictor, classes)
        results_dict = {}
        for camera_name in request.files.keys():
            results_dict[camera_name + "_masks"] = np.empty(shape=(0, 0))
            results_dict[camera_name + "_boxes"] = np.empty(shape=(0, 0))
            results_dict[camera_name + "_classes"] = np.empty(shape=(0, 0))
            results_dict[camera_name + "_scores"] = np.empty(shape=(0, 0))
        # Read the received images, which are each associated
        # with a different key corresponding to the name of
        # the camera the image was taken from.
        for camera_name, img_file in request.files.items():
            img_bytes = img_file.read()
            try:
                image = read_image(img_bytes)
            except (PIL.UnidentifiedImageError, ValueError) as e:
                print(f"Error with image from camera {camera_name}: {e}")
                continue
            # Query Detic with this particular image and get the output.
            boxes, class_idx, scores = Detic(image, detic_predictor)
            if len(boxes) == 0:
                continue
            masks = SAM(image, boxes, sam_predictor)
            masks = masks.cpu().numpy()
            classes = [classes[idx] for idx in class_idx]
            results_dict[camera_name + "_masks"] = masks
            results_dict[camera_name + "_boxes"] = boxes
            results_dict[camera_name + "_classes"] = classes
            results_dict[camera_name + "_scores"] = scores

        # Send result.
        buf = io.BytesIO()
        np.savez(buf, **results_dict)
        buf.seek(0)
        return send_file(buf, mimetype="numpy", as_attachment=True, download_name="result.npy")

if __name__ == '__main__':
    app.run(port=5550)
