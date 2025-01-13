import cv2
import numpy as np

def main():
    # 1. Initialize your camera or video file
    #    If you have a USB camera connected, it is often index 0 or 1.
    #    Or replace '0' with a path to a video file: e.g., 'test_video.mp4'
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    # 2. Create a background subtractor
    #    MOG2 is one of the standard background subtraction algorithms in OpenCV.
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=True)

    while True:
        # 3. Read the current frame
        ret, frame = cap.read()
        if not ret:
            break

        # 4. Apply the background subtractor
        fg_mask = back_sub.apply(frame)

        # 5. (Optional) Morphological operations to reduce noise. Don't need it tho.
        # kernel = np.ones((5, 5), np.uint8)
        # fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        # fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        # 6. Threshold the mask to get a clear binary image
        #    This step can help remove intermediate gray pixels from shadow detection
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # 7. Find contours on the foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 8. Draw bounding boxes around detected objects
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Ignore small or noisy contours by an area threshold
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 9. Display the results
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Foreground Mask', fg_mask)

        # 10. Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
