import cv2

def main():
    cap = cv2.VideoCapture(0)
    num = 0
    while cap.isOpened():
        ret, frames = cap.read()
        k = cv2.waitKey(5)  # wait for 5 milliseconds

        if k == 27: # if "Esc" value is pressed, then exit
            break
        elif k == ord('s'):  # wait for 's' key to save and exit
            cv2.imwrite('calibration_images/img' + str(num) + '.png', frames)
            print("image saved!")
            num += 1

        cv2.imshow('Image', frames)

    # Release and destroy all windows before termination
    cap.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
