import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    # Create a background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fg_mask = back_sub.apply(frame)

        # Morphological operations to reduce noise. Don't need it tho.
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        # _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 250:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Original Frame', frame)
        cv2.imshow('Foreground Mask', fg_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
