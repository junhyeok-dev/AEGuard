import cv2
import sys
import numpy as np
from scipy.stats import entropy

img = None

WINDOW_SIZE = 7
STEP = 2

is_adv = 1

if len(sys.argv) != 2:
    print("Usage: python ImageAnalyzer.py [FILENAME]")
    exit(1)
else:
    img = cv2.imread(sys.argv[1])
    if sys.argv[1].startswith("adv"):
        is_adv = 1

h, w = len(img), len(img[0])
gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


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


def edgeNoiseAnalysis(image):
    edge = cv2.Canny(image, 50, 100)

    base = cv2.getGaussianKernel(3, 3)
    kernel = np.outer(base, base.transpose())

    arr = cv2.filter2D(edge, -1, kernel)

    edgecount = 0
    arrcount = 0

    for i in range(w):
        for j in range(h):
            if arr[i][j] < 85:
                arr[i][j] = 0
            else:
                arr[i][j] = 255
                arrcount += 1

            if edge[i][j] == 255:
                edgecount += 1

    return arrcount - edgecount


def backgroundDensityAnalysis(image):

    return 0

'''
print("Total entropy: ", entropy(entropy(gs)))
print("Total variance: ", totalVariance(gs))
print("Edge density: ", edgeDensityAnalysis(img))
print("Color composition (b, g, r): ", colorCompositionAnalysis(img))
print("Edge diff after gaussian filtering: ", edgeNoiseAnalysis(img))
'''

cc = colorCompositionAnalysis(img)

f = open('analysis_result.csv', 'a')
f.write(
    str(entropy(entropy(gs))) + ',' + str(totalVariance(gs)) + ',' +
    str(edgeDensityAnalysis(img)) + ',' + str(cc[0]) + ',' + str(cc[1]) + ',' +
    str(cc[2]) + ',' + str(edgeNoiseAnalysis(img)) + ',' + str(is_adv) + '\n'
)