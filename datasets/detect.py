import os
from ultralytics import YOLO
import argparse


model = YOLO('runs/detect/yolov8n_UAVs/weights/best.pt')


# Detects using YOLO's predict
def main():

    parser = argparse.ArgumentParser(description="YOLO image and character detection")
    parser.add_argument("file", help="Path to the image file or directory containing .png images of ODLC")

    args = parser.parse_args()

    image_path = args.file

    if not os.path.exists(image_path):
        print(f"Error: The path '{image_path}' does not exist.")
        return
    

    results = model.predict(
        source=image_path,
        imgsz=416,
        conf=0.10,
        save=False,
        verbose=False
    )

   # Write to vision.out
    with open("vision.out", "w") as output_file:
        for idx, result in enumerate(results):
            output_file.write(f"\nImage {idx + 1}:\n")
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = box.conf[0].item()
                xyxy = box.xyxy[0].tolist()
                class_name = model.names[cls_id]

                output_file.write(f"{class_name}")
                print(f" - Detected {class_name} with confidence {confidence:.2f} at {xyxy}")  # Print helps with visualization



if __name__ == "__main__":
    main()