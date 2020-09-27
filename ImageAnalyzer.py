import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np

img = None

if len(sys.argv) != 2:
    print("Usage: python ImageAnalyzer.py [FILENAME]")
    exit(1)
else:
    img = cv2.imread(sys.argv[1])



def EdgeDensityAnalysis(image):
    edge = cv2.Canny(image, 50, 100)
    print(edge)
    plt.imshow(edge, cmap='gray')
    plt.show()
    return edge


def ColorCompositionAnalysis(image):
    b, g, r = cv2.split(image)


def BackgroundDensityAnalysis(image):

    return 0


EdgeDensityAnalysis(img)