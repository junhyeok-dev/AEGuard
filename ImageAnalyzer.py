import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np

img = None

WINDOW_SIZE = 7
STEP = 2

if len(sys.argv) != 2:
    print("Usage: python ImageAnalyzer.py [FILENAME]")
    exit(1)
else:
    img = cv2.imread(sys.argv[1])

h, w = len(img), len(img[0])


def __calcMean(image):
    pixel_sum = 0
    size = 0
    for row in image:
        for pixel in row:
            pixel_sum += pixel
            size += 1

    return pixel_sum / size


def totalVariance(image):
    mean = __calcMean(image)
    sum_distance = 0
    size = 0
    for row in image:
        for pixel in row:
            sum_distance += ((pixel - mean) ** 2)
            size += 1

    return sum_distance / size


def edgeDensityAnalysis(image):
    edge = cv2.Canny(image, 50, 100)
    plt.imshow(edge,cmap="Greys")
    plt.show()

    total_slc = 0
    slc_include_edge = 0

    for i in range(w):
        for j in range(h):
            if edge[i][j] == 255:
                slc_include_edge += 1
            total_slc += 1

    return (slc_include_edge / total_slc) * 100


def colorCompositionAnalysis(image):
    total = [0, 0, 0]
    b, g, r = cv2.split(image)
    for i in range(w):
        for j in range(h):
            total[0] += b[i][j]
            total[1] += g[i][j]
            total[2] += r[i][j]

    return total


def backgroundDensityAnalysis(image):

    return 0


print("전체 분산: ", totalVariance(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
print("엣지 밀도: ", edgeDensityAnalysis(img))
print("색상 구성: ", colorCompositionAnalysis(img))