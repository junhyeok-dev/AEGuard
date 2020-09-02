import tensorflow as tf

model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
model.trainable = False

model = tf.keras.applications.ResNet152V2

decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

def preprocess(image):
    img = tf.cast(image, tf.float32)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = img[None, ...]

    return img

def get_imagenet_label(probs):
    return decode_predictions(probs, top=1)[0][0]

img_path = tf.keras.utils.get_file()

