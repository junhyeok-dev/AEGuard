import keras
import sys
from joblib import load
import module.imganalyze as imganalyze
import cv2

def predict(fname):
    image = cv2.imread(fname)
    print("Selected image:", fname)
    print("Image array:", image)

    try:
        model = keras.models.load_model('AEGuard.h5')
    except OSError:
        print("Error: AEGuard.keras not found")
        exit(2)

    x = []

    try:
        x.append(imganalyze.totalEntropy(image))
        x.append(imganalyze.totalVariance(image))
        x.append(imganalyze.edgeDensityAnalysis(image))
        cc = imganalyze.colorCompositionAnalysis(image)
        x.append(cc[0])
        x.append(cc[1])
        x.append(cc[2])
        x.append(imganalyze.edgeNoiseAnalysis(image, 50, 100))
        x.append(imganalyze.edgeEntropy(image))
    except cv2.error:
        print("Error: Invalid image source")
        exit(3)

    x = [x]

    print('Prediction:', model.predict(x)[0][0])

    result = model.predict(x)[0][0]

    res_txt = ""

    if result < 0.5:
        res_txt = fname + ' is a normal image'
    else:
        res_txt = fname + ' is an adversarial sample'

    print(res_txt)

    return res_txt

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 AEGuard.py [FILENAME]")
        exit(1)
    else:
        fname = sys.argv[1]
        res_txt = predict(fname)
        print(res_txt)