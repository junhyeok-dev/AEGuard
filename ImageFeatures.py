import cv2
import numpy as np
import skimage.measure
import matplotlib.pyplot as plt
import time


def mean2d(image):
    pixel_sum = 0
    size = 0
    for row in image:
        for pixel in row:
            pixel_sum += pixel
            size += 1

    return pixel_sum / size


def dct2d(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.float32(image) / 255.0
    dct = cv2.dct(image)
    image = np.uint8(dct * 255.0)

    return image


def entropy(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return skimage.measure.shannon_entropy(image)


def variance(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = mean2d(image)
    sum_distance = 0
    size = 0
    for row in image:
        for pixel in row:
            sum_distance += ((pixel - mean) ** 2)
            size += 1

    return sum_distance / size


def colorCompose(image):
    h, w = len(image), len(image[0])

    total = [0, 0, 0]
    b, g, r = cv2.split(image)
    for i in range(w):
        for j in range(h):
            total[0] += b[i][j]
            total[1] += g[i][j]
            total[2] += r[i][j]

    total_pix = sum(total)
    total = [val / total_pix * 100 for val in total]

    return total


class edge():
    @staticmethod
    def __autoCanny(image, sigma=0.20, verbose=True):
        med = np.median(image)
        l = int(max(0, (1.0 - sigma) * med))
        u = int(min(255, (1.0 + sigma) * med))

        e = cv2.Canny(image, l, u)

        if verbose:
            print("l:", l, "u:", u)
            plt.imshow(e)
            plt.show()

        return e

    @staticmethod
    def density(image):
        h, w = len(image), len(image[0])

        e = edge.__autoCanny(image)

        total_slc = 0
        slc_include_edge = 0

        for i in range(w):
            for j in range(h):
                if e[i][j] == 255:
                    slc_include_edge += 1
                total_slc += 1

        return (slc_include_edge / total_slc) * 100

    @staticmethod
    def noise(image):
        h, w = len(image), len(image[0])

        e = edge.__autoCanny(image)
        plt.imshow(e)
        plt.show()

        base = cv2.getGaussianKernel(5, 5)
        kernel = np.outer(base, base.transpose())

        arr = cv2.filter2D(e, -1, kernel)
        plt.imshow(arr)
        plt.show()

        edgecount = 0.0
        arrcount = 0.0

        for i in range(w):
            for j in range(h):
                if arr[i][j] > 55:
                    arrcount += 1
                    arr[i][j] = 255
                else:
                    arr[i][j] = 0

                if e[i][j] == 255:
                    edgecount += 1

        plt.imshow(arr)
        plt.show()

        return (arrcount - edgecount) / edgecount * 100

    @staticmethod
    def gradient(image, slice_size, steps=1, verbose=True):
        h, w = len(image), len(image[0])
        half_slice_size = int(slice_size / 2)

        e = edge.__autoCanny(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        start = time.time()

        cnt = 0
        ent = 0

        for i in range(half_slice_size, h - half_slice_size, steps):
            for j in range(half_slice_size, w - half_slice_size, steps):
                if e[i][j] == 255:
                    l, r, u, b = j - half_slice_size, j + half_slice_size, i - half_slice_size, i + half_slice_size
                    slice_e = e[u:b, l:r]
                    slice_o = image[u:b, l:r]

                    #plt.imshow(slice_e);plt.show()
                    #plt.imshow(slice_o);plt.show()

                    #print(slice_o)
                    #print(slice_e)

                    diff_h_sum = 0
                    diff_h_count = 0

                    for si in range(1, slice_size - 1):
                        for sj in range(1, slice_size - 1):
                            if slice_e[si][sj] == 255:
                                diff_h = abs(slice_o[si][sj - 1] - slice_o[si][sj + 1])
                                #print("diff:", diff_h)

                                diff_h_sum += diff_h
                                diff_h_count += 1

                    ent += (diff_h_sum / diff_h_count)
                    cnt += 1



        if verbose:
            print("CPU Time:", time.time() - start)
            print("Edge diff:", ent / cnt)


    @staticmethod
    def entropy(image):
        e = edge.__autoCanny(image)
        return skimage.measure.shannon_entropy(e)

    @staticmethod
    def visualize(image, title):
        e = edge.__autoCanny(image)

        plt.title(title)
        plt.imshow(e)
        plt.show()
