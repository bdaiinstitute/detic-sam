# Open Vocabulary Object Detection and Segmentation with Detic and Segment Anything

This repo uses [Detic](https://github.com/facebookresearch/Detic) to detect objects based on a text description of each class (i.e. open-vocabulary detection). It then conditions the [Segment Anything](https://github.com/facebookresearch/segment-anything) model with the detected bounding boxes to get segmentation masks.

<p float="left">
  <img src="examples/1_segm.png"  width="300">
  <img src="examples/2_segm.png"  width="300">
</p>

## Setup

First add the following to your bash profile (assuming you have CUDA 11+):
`export CUDA_PATH=/usr/local/cuda-11.7/`

Next, be sure to use `python 3.8`. If you have a higher version of python, then install 3.8 and use this.

Either run `./setup.sh` (make sure the `python` command uses python 3.8 in this case!) or follow the steps manually.

## Usage

Segmenting example images `1.png` and `2.png`.
```
source venv/bin/activate
python main.py 1.png -c bottle mug spoon "mug rack" box cpu bowl -d "cuda:0"
python main.py 2.png -c screwdriver "scrubbing brush" -d "cuda:0"
```

## Troubleshooting
- Segmentation Fault (core dumped) as soon as you run `server.py`
  This stems from an issue with the detectron2 installation. Torch and detectron2 are closely linked: you need to make
  sure you've installed the torch version with the right CUDA extension corresponding to the detectron2 version (as well
  as your own system setup). Find the correct detectron2 installation command [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
  Then, find the corresponding torch version compatible withthat and make sure you have that.
- `PIL.Image.LINEAR` doesn't exist.
  If you see something like the below when trying to run `server.py`:
  ```
    File "/home/nkumar/detic-sam/venv/lib/python3.10/site-packages/detectron2/data/transforms/transform.py", line 46, in ExtentTransform
    def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):
    AttributeError: module 'PIL.Image' has no attribute 'LINEAR'. Did you mean: 'BILINEAR'?
  ```
  Then simply edit the offending file to change `Image.LINEAR` to `Image.BILINEAR`.

## Licenses and Acks

This code is based on [prediction_in_wild](https://github.com/pagidik/prediction_in_wild), which was made by Kishore Pagidi and is licensed under MIT.
His repository in turn uses [detectron2](https://github.com/facebookresearch/detectron2), [Detic](https://github.com/facebookresearch/Detic) and [SAM](https://github.com/facebookresearch/segment-anything), which are licensed under Apache-2.0.
