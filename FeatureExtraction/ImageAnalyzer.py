import cv2
import sys
import ImageFeatures as imganalyze

img = None

is_adv = 0

if len(sys.argv) != 2:
    print("Usage: python ImageAnalyzer.py [FILENAME]")
    exit(1)
else:
    img = cv2.imread(sys.argv[1])
    if sys.argv[1].startswith("./dataset/cifar100/adv"):
        is_adv = 1

cc = imganalyze.colorCompose(img)

f = open('analysis_result.csv', 'a')
f.write(
    str(imganalyze.entropy(img)) + ',' + str(imganalyze.variance(img)) + ',' +
    str(imganalyze.edgeDensityAnalysis(img)) + ',' + str(cc[0]) + ',' + str(cc[1]) + ',' +
    str(cc[2]) + ',' + str(imganalyze.edgeNoiseAnalysis(img, 50, 100)) + ',' + str(imganalyze.edgeEntropy(img)) + ',' + str(is_adv) + '\n'
)