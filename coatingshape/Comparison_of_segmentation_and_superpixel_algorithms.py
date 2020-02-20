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
from skimage.color import label2rgb

from skimage import exposure
from scipy.ndimage import gaussian_filter
from skimage import data
from skimage import img_as_float
from skimage.morphology import reconstruction
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries

import matplotlib
#Data1 = io.imread('coatingshape/x1_side_mid.tif', plugin='tifffile')
Data1 = io.imread('coatingshape/1.tif', as_gray=True)
Data2 = io.imread('coatingshape/1.tif', as_gray=True)
#Data1[Data1<120] = 255
# Data1 = Data1[200:350, :]
img = img_as_float(Data2)

segments_fz = felzenszwalb(img, scale=150, sigma=0.6, min_size=50)
segments_slic = slic(img, n_segments=500, compactness=10, sigma=1)
#segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
gradient = sobel(rgb2gray(img))
segments_watershed = watershed(gradient, markers=250, compactness=0.1)



print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
print(f"SLIC number of segments: {len(np.unique(segments_slic))}")
#print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(img, segments_slic))
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()