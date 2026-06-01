import argparse
import time
from pathlib import Path

import cv2
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


def live_detect():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: could not open webcam")
        return

    print("Live detection running — press Q to quit")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=[0], verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"person {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        count = len(results.boxes)
        cv2.putText(frame, f"People: {count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        cv2.imshow("Live Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run live webcam detection")
    args = parser.parse_args()

    if args.live:
        live_detect()
    else:
        detect_all()