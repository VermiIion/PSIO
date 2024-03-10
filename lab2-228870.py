import skimage
import numpy as np
import matplotlib.pyplot as plt
import collections

moon = skimage.data.moon()

hist = plt.hist(moon,range=(0,255),bins=256)
plt.subplot(121),plt.imshow()
plt.show()
