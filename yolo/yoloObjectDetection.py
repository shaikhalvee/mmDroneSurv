import cv2
from ultralytics import YOLO

def main():
    # 1. Initialize the YOLO model
    #    This loads a pretrained YOLOv8 model trained on the COCO dataset.
    #    If you're detecting drones specifically, you would replace 'yolov8n.pt'
    #    with your own custom-trained weights (e.g., 'best.pt').
    model = YOLO('yolov8n.pt')

    # 2. Open a connection to your webcam (source=0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # 3. Use the YOLO model to perform object detection
        #    By default, this returns a list of results for each image in the batch.
        results = model.predict(frame, imgsz=640, conf=0.5)
        #   - 'imgsz=640' sets the input image size for inference.
        #   - 'conf=0.5' sets the confidence threshold (0.0 ~ 1.0).

        # 4. YOLOv8 returns a "Results" object. We'll grab the "plot" version
        #    which has bounding boxes and labels drawn on the frame.
        annotated_frame = results[0].plot()

        # 5. Display the annotated frame
        cv2.imshow('YOLOv8 Detection', annotated_frame)

        # 6. Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
