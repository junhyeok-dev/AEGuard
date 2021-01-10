import keras
import sys
from sklearn.preprocessing import StandardScaler
from joblib import load
import module.imganalyze as imganalyze
import cv2

model = None
FILENAME = 'test.png'
image = None

if len(sys.argv) != 2:
    print("Usage: python3 AEGuard.py [FILENAME]")
    exit(1)
else:
    FILENAME = sys.argv[1]

image = cv2.imread(FILENAME)

try:
    model = keras.models.load_model('AEGuard.keras')
except OSError:
    print("Error: AEGuard.keras not found")
    exit(2)

scaler = load('Scaler.bin')

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

x = scaler.fit_transform(x)

print('Prediction:', model.predict(x)[0][0])

result = model.predict(x)[0][0]

if result < 0.5:
    print(FILENAME, 'is a normal image')
else:
    print(FILENAME, 'is an adversarial sample')