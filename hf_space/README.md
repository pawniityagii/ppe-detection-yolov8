---
title: PPE Safety Helmet Detection
emoji: 🦺
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# PPE / Safety-Helmet Detection (YOLOv8)

A live object-detection demo: upload a workplace image and the model draws
boxes around safety helmets (hard hats), with a confidence slider.

Trained with YOLOv8 on a public hard-hat/PPE dataset. See the training and
evaluation code at the GitHub repo:
https://github.com/pawniityagii/ppe-detection-yolov8

## Files in this Space
- `app.py` - the Gradio app (entry point)
- `requirements.txt` - pinned dependencies
- `weights/best.pt` - the trained model. Upload yours here after training.

If `weights/best.pt` is missing, the app falls back to the pretrained COCO
`yolov8n` model so the Space still launches.
