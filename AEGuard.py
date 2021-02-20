import keras
import sys
from FeatureExtraction.modules import ImageFeatures
import cv2


def predict(fname):
    image = cv2.imread(fname)

    image = cv2.resize(image, (224, 224))

    try:
        model = keras.models.load_model('AEGuard.keras')
    except OSError:
        print("Error: AEGuard.keras not found")
        exit(2)

    x = []

    try:
        x.append(ImageFeatures.entropy(image))
        x.append(ImageFeatures.variance(image))
        x.append(ImageFeatures.dctBias2d(image))
        cc = ImageFeatures.colorCompose(image)
        x.append(ImageFeatures.edge.density(image))
        x.append(cc[0])
        x.append(cc[1])
        x.append(cc[2])
        x.append(ImageFeatures.edge.noise(image))
        x.append(ImageFeatures.edge.entropy(image))
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