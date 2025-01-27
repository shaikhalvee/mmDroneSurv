import cv2
import numpy as np

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use the camera index (0 for default camera)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Initialize the background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    while True:
        # Capture each frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize frame for faster processing (optional)
        # frame_resized = cv2.resize(frame, (640, 480))
        frame_resized = frame

        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame_resized)

        # Get height and width of the frames
        orig_frame_height, orig_frame_width = frame_resized.shape[:2]
        mask_frame_height, mask_frame_width = fg_mask.shape[:2]
        print(orig_frame_height, orig_frame_width, mask_frame_height, mask_frame_width)

        # Clean up the mask using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Find contours of the objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around detected objects
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter out small objects
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Center of the bounding box
                b_x = x + w // 2
                b_y = y + h // 2

                # Draw a small circle at the center
                cv2.circle(frame, (int(b_x), int(b_y)), 5, (0, 0, 255), -1)


        # Display the results
        cv2.imshow("Detected Objects", frame_resized)
        cv2.imshow("Foreground Mask", fg_mask)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
