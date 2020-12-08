import cv2
import numpy as np
import skimage.measure


def __calcMean(image):
    pixel_sum = 0
    size = 0
    for row in image:
        for pixel in row:
            pixel_sum += pixel
            size += 1

    return pixel_sum / size


def totalEntropy(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return skimage.measure.shannon_entropy(image)


def totalVariance(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = __calcMean(image)
    sum_distance = 0
    size = 0
    for row in image:
        for pixel in row:
            sum_distance += ((pixel - mean) ** 2)
            size += 1

    return sum_distance / size


def edgeDensityAnalysis(image):
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


def colorCompositionAnalysis(image):
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


def edgeNoiseAnalysis(image):
    h, w = len(image), len(image[0])

    edge = cv2.Canny(image, 50, 100)

    base = cv2.getGaussianKernel(5, 5)
    kernel = np.outer(base, base.transpose())

    arr = cv2.filter2D(edge, -1, kernel)

    edgecount = 0
    arrcount = 0

    for i in range(w):
        for j in range(h):
            if arr[i][j] < 55:
                arr[i][j] = 0
            else:
                arr[i][j] = 255
                arrcount += 1

            if edge[i][j] == 255:
                edgecount += 1

    return (arrcount - edgecount) / edgecount * 100


def edge_entropy(image):
    edge = cv2.Canny(image, 50, 100)
    return skimage.measure.shannon_entropy(edge)