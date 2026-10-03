"""
Hugging Face Spaces app: PPE (safety-helmet) detection.

Upload an image, get it back with bounding boxes drawn around detected
safety helmets (hard hats). The model has a single class: helmet.

This file is the entry point a Hugging Face Space runs. Put best.pt in the
same folder (as weights/best.pt) and this loads it at startup.
"""
import os

import gradio as gr
from ultralytics import YOLO

WEIGHTS = os.environ.get("WEIGHTS", "weights/best.pt")

# Fall back to the pretrained COCO nano model if trained weights are absent,
# so the Space still launches while you finish training. Replace with best.pt
# for the real PPE model.
model = YOLO(WEIGHTS if os.path.exists(WEIGHTS) else "yolov8n.pt")


def detect(image, conf):
    """Run detection and return the annotated image plus a text summary."""
    if image is None:
        return None, "Upload an image to run detection."
    results = model.predict(source=image, conf=conf, imgsz=640, verbose=False)
    r = results[0]
    annotated = r.plot()[:, :, ::-1]  # BGR (OpenCV) -> RGB for display
    counts = {}
    for c in r.boxes.cls.tolist():
        name = r.names[int(c)]
        counts[name] = counts.get(name, 0) + 1
    summary = ", ".join(f"{v} x {k}" for k, v in counts.items()) or "No detections."
    return annotated, summary


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="numpy", label="Input image"),
        gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="Confidence threshold"),
    ],
    outputs=[
        gr.Image(type="numpy", label="Detections"),
        gr.Text(label="Summary"),
    ],
    title="PPE / Safety-Helmet Detection (YOLOv8)",
    description=(
        "Object detection model that flags safety helmets (hard hats) in an image. "
        "Trained on a public hard-hat/PPE dataset with YOLOv8. Upload a workplace "
        "photo and adjust the confidence threshold."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
