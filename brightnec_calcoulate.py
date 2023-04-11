import cv2
import numpy as np

#
# def calculate_brightness(image):
#
#     greyscale_image = image.convert('L')
#     histogram = greyscale_image.histogram()
#     pixels = sum(histogram)
#     brightness = scale = len(histogram)
#
#     for index in range(0, scale):
#
#         ratio = histogram[index] / pixels
#         brightness += ratio * (-scale + index)
#     return 1 if brightness == 255 else brightness / scale
#

from numpy.linalg import norm


def brightness(img):
    if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)

if __name__ == '__main__':
    # The function imread loads an image
    # from the specified file and returns it.
    original = cv2.imread("b_w_bug/IMG_0082.JPG")
    bright = brightness(original)
    print(int(bright))

    # Making another copy of an image.
    img = original.copy()

