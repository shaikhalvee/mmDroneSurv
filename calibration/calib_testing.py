import cv2


def main():
    chessboard_size = (9, 6)
    fname = "calibration_images/img2.png"
    img = cv2.imread(fname)
    cv2.imshow("Test Image", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        print(f"Chessboard found in {fname}")
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(1500)  # Show image with corners drawn
    else:
        print(f"Chessboard not found in {fname}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
