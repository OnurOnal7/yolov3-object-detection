# YOLOv3 Object Detection with Darknet and OpenCV

A Google Colab notebook that demonstrates **object detection with YOLOv3** using a modified **Darknet** implementation, **OpenCV** image preprocessing, and **COCO** class labels.

## Overview

This notebook walks through a simple end-to-end YOLOv3 inference pipeline:

- mounting Google Drive in Colab
- loading project files from Drive
- installing and importing OpenCV and helper modules
- building the YOLOv3 network from `yolov3.cfg`
- loading pretrained `yolov3.weights`
- loading COCO class names
- printing the YOLOv3 / Darknet network structure
- reading and resizing input images
- setting IoU and non-max suppression thresholds
- running object detection on example images
- visualizing detections and inspecting how threshold choices affect results

## Model

The notebook uses **YOLOv3**, a one-stage object detector built on the **Darknet-53** backbone.

Project files referenced in the notebook include:

- `cfg/yolov3.cfg` for network architecture
- `weights/yolov3.weights` for pretrained weights
- `data/coco.names` for the 80 COCO object classes

The network is loaded through a custom `Darknet` class and the pretrained weights are applied before inference.

## Environment

This notebook is written for **Google Colab** and assumes the project folder is stored in **Google Drive**.

The workflow includes:

- mounting Drive at `/content/drive`
- appending the project directory to `sys.path`
- installing `opencv-python`
- importing:
  - `cv2`
  - `matplotlib`
  - `utils`
  - `darknet`

## Image Preprocessing

Images are loaded with OpenCV and converted from **BGR to RGB** for display.

Before inference, each image is resized to the model input size defined by the YOLOv3 configuration. The notebook also displays the original image alongside the resized version to show the preprocessing step clearly.

## Detection Pipeline

The notebook performs detection using a helper function built around YOLOv3 inference.

Key steps include:

- forwarding the resized image through the network
- collecting predicted bounding boxes
- filtering predictions with **Non-Max Suppression**
- removing overlapping detections based on **IoU**
- displaying detected objects with confidence scores

## Threshold Experiments

A main focus of the notebook is understanding how post-processing affects detection quality.

### Non-Max Suppression threshold

The notebook explains and tests the **NMS threshold**, which removes boxes with low detection probability.

Example value used:

- `nms_thresh = 0.6`

### Intersection over Union threshold

The notebook also explores the **IoU threshold**, which controls when overlapping boxes should be suppressed.

Example values used:

- `iou_thresh = 0.4`
- later experiments also use `iou_thresh = 0.2` and `iou_thresh = 0.6`

These experiments show how detection outputs change depending on the amount of filtering applied.

## Example Outputs

The notebook runs inference on example images such as:

- `images/brockp.jpeg`
- `images/VOC9955.jpg`

For the `VOC9955.jpg` example shown in the outputs, the detector reports:

- **runtime:** about **1.718 seconds**
- **objects detected:** **3**

The listed detections include **boat** predictions with high confidence, illustrating successful COCO-class inference with YOLOv3.

## What this notebook demonstrates

This project is mainly an **inference and visualization notebook**, not a training pipeline.

It is useful for:

- understanding how YOLOv3 is loaded from config and weight files
- seeing how images are prepared for object detection
- learning the role of IoU and NMS in post-processing
- observing how threshold settings affect final detections

## Requirements

Main dependencies used in the notebook:

    pip install opencv-python matplotlib numpy

## Project Structure

The notebook expects a project layout similar to:

    cfg/
    ├── yolov3.cfg

    weights/
    ├── yolov3.weights

    data/
    ├── coco.names

    images/
    ├── brockp.jpeg
    └── VOC9955.jpg

    darknet.py
    utils.py
    yolov3_object_detection.ipynb

## Notes

- The notebook is designed for **Google Colab + Google Drive** paths.
- It uses a custom `darknet.py` and `utils.py` rather than the official Darknet repository directly.
- The workflow focuses on **object detection inference**, threshold tuning, and visualization.

## Summary

This notebook is a compact YOLOv3 object detection demo that shows how to load a pretrained Darknet model, preprocess images, run inference, and study the effects of **IoU** and **non-max suppression** on final detections.

## Note on missing weights

The notebook expects the file `weights/yolov3.weights`, but that weights file is **not included here**. To run the notebook successfully, you will need to place the pretrained YOLOv3 weights file at:

    weights/yolov3.weights
