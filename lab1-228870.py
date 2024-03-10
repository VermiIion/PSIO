import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
import cv2

# 1

arr = np.array([[1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [20, 21, 22, 23, 24],
                [25, 26, 27, 28, 29],
                [11, 12, 13, 14, 15],
                [11, 12, 13, 14, 15],
                [11, 12, 13, 14, 15]])  # 9x5


def odbicieWPionie(arr):
    return arr[::-1, ::]


def lustrzaneOdbicie(arr):
    return arr[::, ::-1]


def obrot90prawo(arr):
    return arr[::, ::-1]


def obrot90lewo(arr):
    return arr[::-1, ::]


def obrot180(arr):
    return arr[::-1, ::-1]


def rozszerzenie(arr):
    M, N = arr.shape
    if M <= N:
        return arr
    arr2 = np.zeros((M, M))
    arr2[0:M, ((M - N) // 2):((M - N) // 2 + N)] = arr
    return arr2


def wyciecie(arr):
    M, N = arr.shape
    if M <= N:
        return arr
    arr2 = np.zeros((N, N))
    arr2[0:N // 2, ::] = arr[0:N // 2, ::]
    arr2[N // 2:N, ::] = arr[(M - N + (N // 2)):M, ::]
    return arr2


print(arr)
print("odbicie w pionie")
print(odbicieWPionie(arr))
print("lustrzane odbicie")
print(odbicieWPionie(arr))
print("obrot o 90 stopni w prawo")
print(obrot90prawo(arr))
print("obrot o 90 stopni w lewo")
print(obrot90lewo(arr))
print("obrot 180 stopni")
print(obrot180(arr))
print("rozszerzenie")
print(rozszerzenie(arr))
print("wyciecie")
print(wyciecie(arr))

# 2

img = io.imread("input1/lena.png")
gray = color.rgb2gray(img)
plt.imshow(gray, cmap='gray')
plt.show()

crop = np.ones((640, 480))

x_offset = (crop.shape[0] - gray.shape[0]) // 2
y_offset = (crop.shape[1] - gray.shape[1]) // 2

crop[x_offset:x_offset + gray.shape[0], y_offset:y_offset + gray.shape[1]] = gray

plt.imshow(crop)
plt.show()

# 3
cuts = int(input("Podaj ilosc elementow na boku: "))

dog = cv2.imread("input1/dog_1.jpg")
dog = cv2.cvtColor(dog, cv2.COLOR_BGR2RGB)

dog_height, dog_width = dog.shape[:2]
min_size = min(dog_height, dog_width)

x_start = (dog_width - min_size) // 2
y_start = (dog_height - min_size) // 2
x_end = x_start + min_size
y_end = y_start + min_size
dog_cut = dog[y_start:y_end, x_start:x_end]

cut_size = dog_cut.shape[0] // cuts

sub_images = [
    dog_cut[
    y * cut_size: (y + 1) * cut_size,
    x * cut_size: (x + 1) * cut_size,
    ]
    for y in range(cuts)
    for x in range(cuts)
]

np.random.shuffle(sub_images)

combined_dog = np.vstack(
    [np.hstack(row) for row in np.array_split(sub_images, cuts)]
)

plt.imshow(combined_dog)
plt.axis("off")
plt.show()
