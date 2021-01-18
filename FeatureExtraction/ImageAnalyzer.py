import cv2
import sys
import ImageFeatures

img = None

is_adv = 0

if len(sys.argv) != 2:
    print("Usage: python ImageAnalyzer.py [FILENAME]")
    exit(1)
else:
    img = cv2.imread(sys.argv[1])
    if sys.argv[1].startswith("../dataset/train/adv"):
        is_adv = 1

cc = ImageFeatures.colorCompose(img)

f = open('analysis_result.csv', 'a')
f.write(
    str(ImageFeatures.entropy(img)) + ',' + str(ImageFeatures.variance(img)) + ',' +
    str(ImageFeatures.edge.density(img)) + ',' + str(cc[0]) + ',' + str(cc[1]) + ',' +
    str(cc[2]) + ',' + str(ImageFeatures.edge.noise(img)) + ',' + str(ImageFeatures.edge.entropy(img)) + ',' + str(is_adv) + '\n'
)