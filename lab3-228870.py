import matplotlib.pyplot as plt
import numpy as np
import skimage

#1
image = skimage.io.imread("gears1.png")
image = skimage.color.rgb2gray(image)

skimage.filters.try_all_threshold(image, figsize=(10, 10), verbose=False)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

otsu_threshold = skimage.filters.threshold_otsu(image)
binary_otsu = image > otsu_threshold

mean_threshold = skimage.filters.threshold_mean(image)
binary_mean = image > mean_threshold

bins = 256

histogram = skimage.filters.gaussian(np.histogram(image, bins=bins)[0], sigma=2)

axes[0][0].imshow(binary_otsu, cmap="gray")
axes[0][0].set_title("Threshold otsu")
axes[0][0].axis("off")

axes[0][1].imshow(binary_mean, cmap="gray")
axes[0][1].set_title("Threshold mean")
axes[0][1].axis("off")

axes[1][0].plot(histogram, "black")
axes[1][0].axvline(x=otsu_threshold * bins, color="red")
axes[1][0].set_title("Histogram (Otsu)")

axes[1][1].plot(histogram, "black")
axes[1][1].axvline(x=mean_threshold * bins, color="red")
axes[1][1].set_title("Histogram (Mean)")

plt.show()
#2
image = skimage.io.imread("printed_text.png")
image = skimage.color.rgb2gray(image)
image = skimage.util.img_as_ubyte(image)

fig, axes = plt.subplots(1, 3, figsize=(10, 10))

background = skimage.filters.rank.maximum(image, skimage.morphology.disk(10))
background = skimage.filters.rank.mean(background, skimage.morphology.disk(10))
image2 = background - image

triangle_threshold = skimage.filters.threshold_triangle(image2)
binary_triangle = image2 > triangle_threshold

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(image2, cmap="gray")
axes[1].set_title("Processed")
axes[1].axis("off")

axes[2].imshow(binary_triangle, cmap="gray")
axes[2].set_title("Processed 2")
axes[2].axis("off")

plt.show()
