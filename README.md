# PPE / Safety-Helmet Detection (YOLOv8)

An object-detection model that flags **people and safety helmets (hard hats)** in
workplace images. Built to practise the full computer-vision pipeline: dataset
preparation, training with transfer learning, evaluation with mAP, inference on
new images, and deployment as a live web demo.

**Live demo:** https://huggingface.co/spaces/pawniityagii/ppe-detection (upload an image, see detections)
**Model:** YOLOv8n fine-tuned on a public hard-hat/PPE dataset.

> Fill in your real numbers and links where marked `TODO` after you run the
> training notebook.

## Results

| Metric | Value |
| --- | --- |
| mAP@50 | TODO (from the notebook's evaluate cell) |
| mAP@50-95 | TODO |
| Classes | head, helmet, person |
| Images | TODO (train / val split) |

## What is in here

```
notebooks/train_ppe_yolov8_colab.ipynb   train + evaluate on Colab's free GPU
infer.py                                  run detection locally on images/folders/video
hf_space/                                 the Hugging Face Space (live Gradio demo)
requirements.txt                          local dependencies
sample_images/                            a few images to test inference plumbing
```

## How it works (so you can explain it)

- **Task.** Object detection: for each image the model predicts bounding boxes
  plus a class label and confidence for every object it finds. This is harder
  than image *classification*, which only labels the whole image.
- **Model.** YOLOv8 ("You Only Look Once", version 8) is a single-stage detector:
  one forward pass predicts all boxes at once, which is why it is fast enough for
  real-time and edge use. `n` (nano) is the smallest variant, a sensible default
  on limited hardware.
- **Transfer learning.** Training starts from `yolov8n.pt`, weights already
  trained on the COCO dataset. The model has learned general visual features, so
  fine-tuning on a few thousand PPE images is enough rather than training from
  scratch.
- **Metric.** mAP (mean Average Precision) is the standard detection score.
  `mAP@50` counts a detection correct when its box overlaps the true box by at
  least 50% (IoU 0.5). `mAP@50-95` averages that across stricter overlap
  thresholds, so it is the tougher, more honest number.
- **Confidence threshold.** At inference you keep only detections above a
  confidence value. Raising it cuts false positives but can miss objects; the
  demo exposes this as a slider.

## Reproduce it

1. **Train (Colab).** Open `notebooks/train_ppe_yolov8_colab.ipynb` in Google
   Colab, set the runtime to GPU, and run the cells. It installs YOLOv8, pulls a
   public hard-hat/PPE dataset from Roboflow (free account needed), trains, prints
   mAP, and downloads `best.pt`.
2. **Infer locally.** Put `best.pt` in `weights/`, then:
   ```bash
   pip install -r requirements.txt
   python infer.py --weights weights/best.pt --source sample_images/
   ```
   Annotated images land in `runs/detect/predict/`.
3. **Deploy.** See `hf_space/` - create a free Gradio Space on Hugging Face,
   upload those files plus your `best.pt`, and it goes live at a public URL.

## Notes

- The sample images are only for checking that inference runs; the real
  evaluation is the mAP the notebook reports on the held-out validation set.
- Dataset licences belong to their authors on Roboflow Universe; this repo is the
  code, not the data.
