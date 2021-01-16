import cv2
import numpy as np
import skimage.measure
import matplotlib.pyplot as plt
import threading


def mean2d(image):
    pixel_sum = 0
    size = 0
    for row in image:
        for pixel in row:
            pixel_sum += pixel
            size += 1

    return pixel_sum / size


def dct2d(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.float32(image) / 255.0
    dct = cv2.dct(image)
    image = np.uint8(dct * 255.0)

    return image


def entropy(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return skimage.measure.shannon_entropy(image)


def variance(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = mean2d(image)
    sum_distance = 0
    size = 0
    for row in image:
        for pixel in row:
            sum_distance += ((pixel - mean) ** 2)
            size += 1

    return sum_distance / size


def colorCompose(image):
    h, w = len(image), len(image[0])

    total = [0, 0, 0]
    b, g, r = cv2.split(image)
    for i in range(w):
        for j in range(h):
            total[0] += b[i][j]
            total[1] += g[i][j]
            total[2] += r[i][j]

    total_pix = sum(total)
    total = [val / total_pix * 100 for val in total]

    return total


class edge():
    def density(self, image):
        h, w = len(image), len(image[0])

        edge = cv2.Canny(image, 50, 100)

        total_slc = 0
        slc_include_edge = 0

        for i in range(w):
            for j in range(h):
                if edge[i][j] == 255:
                    slc_include_edge += 1
                total_slc += 1

        return (slc_include_edge / total_slc) * 100


    def noise(self, image, p1, p2):
        h, w = len(image), len(image[0])

        edge = cv2.Canny(image, p1, p2)

        base = cv2.getGaussianKernel(5, 5)
        kernel = np.outer(base, base.transpose())

        arr = cv2.filter2D(edge, -1, kernel)

        edgecount = 0.0
        arrcount = 0.0

        for i in range(w):
            for j in range(h):
                if arr[i][j] > 55:
                    arrcount += 1

                if edge[i][j] == 255:
                    edgecount += 1

        return (arrcount - edgecount) / edgecount * 100


    def gradient(self, image, slice_size):
        h, w = len(image), len(image[0])

        edge = cv2.Canny(image, 50, 100)

        for i in range(0, )

    def entropy(self, image):
        edge = cv2.Canny(image, 50, 100)
        return skimage.measure.shannon_entropy(edge)


    def visualize(self, image, p1, p2, title):
        edge = cv2.Canny(image, p1, p2)

        plt.title(title)
        plt.imshow(edge)
        plt.show()


def quantifySharpness():
    return 10
