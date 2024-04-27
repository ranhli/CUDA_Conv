import cv2
import numpy as np
import time

def conv(image, kernel_size=(5, 5), sigma=0):
    output = cv2.GaussianBlur(image, kernel_size, sigma)
    return output

def main():
    # Load an image
    image = cv2.imread('data/1-high-res.png')

    # Define kernel size and sigma
    kernel_size = (5, 5)
    sigma = 0

    # Time the convolution process
    start_time = time.time()
    output = conv(image, kernel_size, sigma)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Convolution time:", elapsed_time, "seconds")

    # Save the blurred image
    cv2.imwrite('output/1-high-res-out-py.png', output)

if __name__ == "__main__":
    main()
