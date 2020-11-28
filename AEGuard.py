import keras
import sys
from sklearn.preprocessing import StandardScaler

model = None

if len(sys.argv) != 2:
    print("Usage: python3 AEGuard.py [FILENAME]")
    exit(1)

try:
    model = keras.models.load_model('AEGuard.keras')
except OSError:
    print("Error: AEGuard.keras not found")
    exit(2)

scaler = StandardScaler()

x = [
    [7.3378496842073835,2005.3647338785904,4.504145408163265,32.902195839142415,33.678372772158376,33.41943138869921,75.08849557522123,0.2649477141230245],
    [7.347049941119418,2043.8005363495167,3.0113998724489797,32.896276164881336,33.6823649646088,33.42135887050986,31.436135009927202,0.19496323402345217]
]

x = scaler.fit_transform(x)

print(model.predict(x))