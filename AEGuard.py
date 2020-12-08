import keras
import sys
from sklearn.preprocessing import StandardScaler
from joblib import load

model = None

if len(sys.argv) != 2:
    print("Usage: python3 AEGuard.py [FILENAME]")
    exit(1)

try:
    model = keras.models.load_model('AEGuard.keras')
except OSError:
    print("Error: AEGuard.keras not found")
    exit(2)

scaler = load('Scaler.bin')

x = [
    [7.075682579440858,2220.102910305168,4.0776466836734695,37.55233154196166,26.82463620955306,35.62303224848528,-2.4926686217008798,0.2458410044201948],
    [7.272445282395836,1612.7371901794559,4.803093112244898,26.2879136301974,35.472294082632224,38.239792287170374,79.75103734439834,0.2779728675194446]
]

x = scaler.fit_transform(x)

print(model.predict(x))