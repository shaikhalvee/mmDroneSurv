import numpy as np
import cv2


def create_chessboard_image(squares_x=10, squares_y=7, square_size=50):
    """
    Creates a checkerboard (chessboard) image with the given number of squares
    horizontally and vertically, and each square having the specified size in pixels.

    squares_x: Number of squares across (10 for a 9x6 internal corner pattern)
    squares_y: Number of squares down  (7 for a 9x6 internal corner pattern)
    square_size: Size of each square in pixels
    """
    # Calculate total image dimensions
    width = squares_x * square_size
    height = squares_y * square_size

    # Initialize a grayscale image (uint8)
    board = np.zeros((height, width), dtype=np.uint8)

    # Fill with alternating white (255) and black (0) squares
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                # White square
                y_start = row * square_size
                y_end = (row + 1) * square_size
                x_start = col * square_size
                x_end = (col + 1) * square_size
                board[y_start:y_end, x_start:x_end] = 255

    return board


def main():
    # For a 9x6 internal corner pattern in OpenCV,
    # you need a 10x7 checkerboard of squares:
    squares_x = 10
    squares_y = 7
    square_size = 5000  # pixels

    checkerboard = create_chessboard_image(squares_x, squares_y, square_size)

    # Save as PNG
    filename = "checkerboard/checkerboard_9x6_corners.png"
    cv2.imwrite(filename, checkerboard)

    print(f"Checkerboard image saved as: {filename}")


if __name__ == "__main__":
    main()
