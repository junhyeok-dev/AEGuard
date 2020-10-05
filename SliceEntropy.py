import cv2
import matplotlib.pyplot as plt
import time
import skimage.measure

# Set constraints
WINDOW_SIZE = 16
STEP = 4
CHANNEL = 0
CHANNEL_TEXT = ['BLUE', 'GREEN', 'RED']

# Load image samples
origin = cv2.imread('images/adv_0.png')
adv = cv2.imread('images/adv_0.01.png')


# Get height and width information from image
h, w = len(origin), len(origin[0])

# Generate grayscale image`
origin_gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
adv_gray = cv2.cvtColor(adv, cv2.COLOR_BGR2GRAY)

# Generate channel split image
origin_channels = cv2.split(origin)
adv_channels = cv2.split(adv)

# Set min/max variables
min_ent = 100
max_ent = 0
min_ch, min_advch = 0, 0
min_i, min_j = 0, 0
max_i, max_j = 0, 0
origin_var, adv_var = 0, 0

mean_diff = 0
total = 0


def calc_mean(image):
    pixel_sum = 0
    size = 0
    for row in image:
        for pixel in row:
            pixel_sum += pixel
            size += 1

    return pixel_sum / size


def amp_func(image, mean, th):
    result = image - int(mean)
    for i in range(len(image)):
        for j in range(len(image[0])):
            if result[i][j] < -th:
                result[i][j] = -th
            elif result[i][j] > th:
                result[i][j] = th

    return result


def calc_var(image):
    mean = calc_mean(image)
    sum_distance = 0
    size = 0
    for row in image:
        for pixel in row:
            sum_distance += ((pixel - mean) ** 2)
            size += 1

    return sum_distance / size


result_file = open('result.csv', 'w')
result_file.write('ent,chent,advent,advchent\n')

start = time.time()
for i in range(0, h - WINDOW_SIZE, STEP):
    for j in range(0, w - WINDOW_SIZE, STEP):
        total += 1  # Count slices

        # Processing on original image slice
        print(str(i), ",", str(j))
        window_origin = origin_gray[i:i + WINDOW_SIZE, j:j + WINDOW_SIZE]
        mean_origin = calc_mean(window_origin)

        # Channel amplification for original image slice
        window_channel = origin_channels[CHANNEL][i:i + WINDOW_SIZE, j:j + WINDOW_SIZE]
        mean_channel = calc_mean(window_channel)
        # window_channel = amp_func(window_channel, mean_channel, 1)

        print("Channel: ", window_channel)

        # Calculate entropy
        origin_ent = skimage.measure.shannon_entropy(window_origin)
        channel_ent = skimage.measure.shannon_entropy(window_channel)

        # Calculate difference between original and amplified channel
        diff = window_origin - window_channel
        diff = sum(diff) / len(diff)
        origin_diff = sum(diff) / len(diff)

        print("origin entropy: ", origin_ent)
        print("origin channel amplified entropy: ", channel_ent)

        result_file.write(str(origin_ent) + ',')
        result_file.write(str(calc_var(window_channel)) + ',')

        # Processing on adversarial image slice
        window_adv = adv_gray[i:i + WINDOW_SIZE, j:j + WINDOW_SIZE]
        mean_adv = calc_mean(window_adv)

        # Channel amplification for adversarial image slice
        window_advch = adv_channels[CHANNEL][i:i + WINDOW_SIZE, j:j + WINDOW_SIZE]
        mean_advch = calc_mean(window_advch)
        # window_advch = amp_func(window_advch, mean_advch, 1)

        print("Adversarial Channel: ", window_advch)

        # Calculate entropy
        adv_ent = skimage.measure.shannon_entropy(window_adv)
        advch_ent = skimage.measure.shannon_entropy(window_advch)

        # Calculate difference between original and amplified channel
        diff = window_adv - window_advch
        diff = sum(diff) / len(diff)
        adv_diff = sum(diff) / len(diff)
        mean_diff += (adv_diff - origin_diff)

        print("adv entropy: ", adv_ent)
        print("adv channel amplified entropy: ", advch_ent)
        print("")

        result_file.write(str(origin_ent) + ',')
        result_file.write(str(calc_var(window_advch)) + '\n')

        if origin_ent < min_ent:
            min_ent = origin_ent
            min_i = i
            min_j = j
            origin_var = calc_var(window_channel)
            adv_var = calc_var(window_advch)

        if origin_ent > max_ent:
            max_ent = origin_ent
            max_i = i
            max_j = j



print("CPU Time: ", time.time() - start)

print("min_entropy: ", min_ent)
print("origin_var: ", origin_var)
print("adv_var: ", adv_var)

origin_rgb = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
adv_rgb = cv2.cvtColor(adv, cv2.COLOR_BGR2RGB)

# Show images on plot
plt.title('Origin')
plt.imshow(origin_rgb[min_i:min_i + WINDOW_SIZE, min_j:min_j + WINDOW_SIZE])
plt.show()
plt.title('Max origin channel: ' + CHANNEL_TEXT[CHANNEL])
plt.imshow(origin_channels[CHANNEL][max_i:max_i + WINDOW_SIZE, max_j:max_j + WINDOW_SIZE], cmap='gray', vmin=0, vmax=255)
plt.show()
plt.title('Min origin channel: ' + CHANNEL_TEXT[CHANNEL])
plt.imshow(origin_channels[CHANNEL][min_i:min_i + WINDOW_SIZE, min_j:min_j + WINDOW_SIZE], cmap='gray', vmin=0, vmax=255)
plt.show()

plt.title('Adversarial')
plt.imshow(adv_rgb[min_i:min_i + WINDOW_SIZE, min_j:min_j + WINDOW_SIZE])
plt.show()
plt.title('Max adversarial channel: ' + CHANNEL_TEXT[CHANNEL])
plt.imshow(adv_channels[CHANNEL][max_i:max_i + WINDOW_SIZE, max_j:max_j + WINDOW_SIZE], cmap='gray', vmin=0, vmax=255)
plt.show()
plt.title('Min adversarial channel: ' + CHANNEL_TEXT[CHANNEL])
plt.imshow(adv_channels[CHANNEL][min_i:min_i + WINDOW_SIZE, min_j:min_j + WINDOW_SIZE], cmap='gray', vmin=0, vmax=255)
plt.show()
