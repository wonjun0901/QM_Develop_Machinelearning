import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage import data, io, filters
from skimage.color import rgba2rgb, rgb2gray
from skimage.external import tifffile
from skimage import feature
from skimage.filters import roberts, sobel, sobel_h, sobel_v
from skimage.filters import threshold_otsu
from skimage import morphology
from scipy import ndimage as ndi

#Data1 = io.imread('coatingshape/x1_side_mid.tif', plugin='tifffile')
Data1 = io.imread('coatingshape/x1_side_mid.tif', as_gray=True)

Data1 = Data1[200:350, :]

#thresh = threshold_otsu(Data1, nbins=50)
#binary = Data1 > thresh


edges1 = roberts(Data1)
edges2 = sobel(Data1)
edges3 = feature.canny(Data1)
fill_edges = ndi.binary_fill_holes(edges2)

markers = np.zeros_like(Data1)
markers[Data1 < 1] = 1
markers[Data1 > 30] = 2

segmentation = morphology.watershed(edges1, markers)


low = 0.03
high = 0.6

lowt = (edges2 > low).astype(int)
hight = (edges2 > high).astype(int)
hyst = filters.apply_hysteresis_threshold(edges2, low, high)


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

ax1.imshow(edges2, cmap=plt.cm.gray)
ax1.axis('off')

ax2.imshow(lowt, cmap=plt.cm.gray)
ax2.axis('off')
plt.show()

#fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
#
#ax1.imshow(edges1, cmap=plt.cm.gray)
#ax1.axis('off')
#
#ax2.imshow(edges2, cmap=plt.cm.gray)
#ax2.axis('off')
#
#ax3.hist(Data1.ravel(), bins=512)
#
#fig.tight_layout()
#plt.show()