import cv2
import numpy as np
import skimage.measure
import matplotlib.pyplot as plt
import threading


def calcMatrixMean(image):
    pixel_sum = 0
    size = 0
    for row in image:
        for pixel in row:
            pixel_sum += pixel
            size += 1

    return pixel_sum / size


def dctCoefficient(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.float32(image) / 255.0
    dct = cv2.dct(image)
    image = np.uint8(dct * 255.0)

    return image


def totalEntropy(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return skimage.measure.shannon_entropy(image)


def totalVariance(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = calcMatrixMean(image)
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


def edgeNoiseAnalysis(image, p1, p2):
    h, w = len(image), len(image[0])

    edge = cv2.Canny(image, p1, p2)

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


def edgeNearbyRise(image):
    edge = cv2.Canny(image, 280, 560)

    edges = []

    for i in range(len(edge)):
        for j in range(len(edge[0])):
            if edge[i][j] == 255:
                edges.append((i, j))

    print(edges)
    plt.imshow(edge)
    plt.show()

    return edge

def edgeEntropy(image):
    edge = cv2.Canny(image, 50, 100)
    return skimage.measure.shannon_entropy(edge)


def VisualizeEdge(image, p1, p2, title):
    edge = cv2.Canny(image, p1, p2)

    plt.title(title)
    plt.imshow(edge)
    plt.show()
