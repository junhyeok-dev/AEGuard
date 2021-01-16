import sys
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
from PIL import Image

if len(sys.argv) != 2:
    print("Usage: python AEGenerator [FILENAME]")
    exit(1)

filename = sys.argv[1]


def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]
    return image


def get_imagenet_label(probs):
    return decode_predictions(probs, top=1)[0][0]


model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
model.trainable = False

decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

image_raw = tf.io.read_file(filename)
image = tf.image.decode_image(image_raw)

filename = filename.split('.')[0]

image = preprocess(image)
image_probs = model.predict(image)

loss_object = tf.keras.losses.CategoricalCrossentropy()


def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction)

    gradient = tape.gradient(loss, input_image)

    signed_grad = tf.sign(gradient)
    return signed_grad


index = 208
label = tf.one_hot(index, image_probs.shape[-1])
label = tf.reshape(label, (1, image_probs.shape[-1]))

perturbations = create_adversarial_pattern(image, label)
plt.imshow(perturbations[0]*0.5+0.5)

epsilons = [0, 0.1, 0.01]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]

_, image_class, class_confidence = get_imagenet_label(image_probs)
print(image_class)

for i, eps in enumerate(epsilons):
    adv_x = image + eps*perturbations
    _, l, c = get_imagenet_label(model.predict(adv_x))
    print(l)
    if eps == 0:
        tf.keras.preprocessing.image.save_img('org_' + filename + '.png', adv_x[0])
    else:
        tf.keras.preprocessing.image.save_img('adv_' + filename + '_' + str(eps) + '.png', adv_x[0])
    adv_x = tf.clip_by_value(adv_x, -1, 1)

