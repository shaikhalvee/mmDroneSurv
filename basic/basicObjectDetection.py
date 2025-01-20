import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    # Create a background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=100, detectShadows=False)

    while True:
        pixel_area_filter_for_contour = 10

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        # Apply background subtraction
        fg_mask = back_sub.apply(frame)

        # Morphological operations to reduce noise. Don't need it tho.
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        # Threshold the mask to remove noise
        _, fg_mask = cv2.threshold(fg_mask, 50, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by size (remove small false positives)
            if area > pixel_area_filter_for_contour:
                x, y, w, h = cv2.boundingRect(contour)
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
