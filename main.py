from pathlib import Path
from ultralytics import YOLO

INPUT_DIR = Path("data/val")
OUTPUT_DIR = Path("data/output")

def detect_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")

    images = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

    if not images:
        print(f"No images found in {INPUT_DIR}")
        return

    for image_path in images:
        output_path = OUTPUT_DIR / (image_path.stem + "_detected" + image_path.suffix)
        results = model(str(image_path))
        result = results[0]
        person_mask = result.boxes.cls == 0
        result.boxes = result.boxes[person_mask]
        result.save(filename=str(output_path))
        count = int(person_mask.sum())
        print(f"Saved: {output_path} ({count} person{'s' if count != 1 else ''} detected)")

if __name__ == "__main__":
    detect_all()