"""
Run PPE detection on an image, a folder of images, or a video.

Usage:
    python infer.py --weights weights/best.pt --source sample_images/
    python infer.py --weights weights/best.pt --source path/to/image.jpg --conf 0.35

Outputs annotated copies to runs/detect/predict/. Nothing is uploaded anywhere.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="PPE detection inference")
    ap.add_argument("--weights", default="weights/best.pt",
                    help="path to trained weights (best.pt from training)")
    ap.add_argument("--source", required=True,
                    help="image file, folder of images, or video file")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="confidence threshold (0-1); raise to cut false positives")
    ap.add_argument("--imgsz", type=int, default=640, help="inference image size")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(
            f"weights not found at {args.weights}. Train first (see the Colab "
            f"notebook) and copy best.pt into weights/."
        )

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        verbose=True,
    )
    # brief summary per image
    for r in results:
        counts = {}
        for c in r.boxes.cls.tolist():
            name = r.names[int(c)]
            counts[name] = counts.get(name, 0) + 1
        print(f"{Path(r.path).name}: {counts or 'no detections'}")
    if results:
        print(f"\nAnnotated outputs saved to: {results[0].save_dir}")


if __name__ == "__main__":
    main()
