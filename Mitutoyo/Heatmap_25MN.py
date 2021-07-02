# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2
import pandas as pd


def makingheatmap(dataname, yaxis_min=720, yaxis_max=820, Num_sample=1):

    if Num_sample == 1:
        # read excel data by pandas
        data = pd.read_excel(dataname)
        # only save information about height of individual microneedle into variable
        height_tiptobase = data.Actual*1000
        print(height_tiptobase)
        # make height information into numpy array
        arrayheight_tiptobase = np.array(height_tiptobase)

        numpy_array_1 = np.zeros([5, 5])
        numpy_array_1[4, :5] = arrayheight_tiptobase[:5]
        numpy_array_1[3, :5] = arrayheight_tiptobase[5:10]
        numpy_array_1[2, :5] = arrayheight_tiptobase[10:15]
        numpy_array_1[1, :5] = arrayheight_tiptobase[15:20]
        numpy_array_1[0, :5] = arrayheight_tiptobase[20:25]

        # Plot the heatmap.
        fig, ax = plt.subplots(figsize=(15, 15))
        im = plt.imshow(numpy_array_1, cmap='Reds', vmin=yaxis_min,
                        vmax=yaxis_max)

        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("height(units:um)", rotation=-90, va="bottom")

        # annotating heatmap by text
        annotate_heatmap(im, valfmt="{x:.1f}")

        return im

    else:
        sample_all_heatmap = np.zeros((1, 97))
        for i in range(Num_sample):
            height_tiptobase = np.array((dataname[i]['Nominal']*1000))
            sample_all_heatmap = height_tiptobase + sample_all_heatmap

        arrayheight_tiptobase = sample_all_heatmap/Num_sample

        numpy_array_1 = np.zeros([11, 11])
        numpy_array_1[10, :3] = None
        numpy_array_1[10, 3:8] = arrayheight_tiptobase[0][:5]  # 1
        numpy_array_1[10, 8:] = None

        numpy_array_1[9, :2] = None
        numpy_array_1[9, 2:9] = arrayheight_tiptobase[0][5:12]  # 2
        numpy_array_1[9, 9:] = None

        numpy_array_1[8, 0] = None
        numpy_array_1[8, 1:10] = arrayheight_tiptobase[0][12:21]  # 3
        numpy_array_1[8, 10] = None

        numpy_array_1[7, 0:11] = arrayheight_tiptobase[0][21:32]  # 4
        numpy_array_1[6, 0:11] = arrayheight_tiptobase[0][32:43]  # 5
        numpy_array_1[5, 0:11] = arrayheight_tiptobase[0][43:54]  # 6
        numpy_array_1[4, 0:11] = arrayheight_tiptobase[0][54:65]  # 7
        numpy_array_1[3, 0:11] = arrayheight_tiptobase[0][65:76]  # 8

        numpy_array_1[2, 0] = None
        numpy_array_1[2, 1:10] = arrayheight_tiptobase[0][76:85]  # 9
        numpy_array_1[2, 10] = None

        numpy_array_1[1, :2] = None
        numpy_array_1[1, 2:9] = arrayheight_tiptobase[0][85:92]  # 10
        numpy_array_1[1, 9:] = None

        numpy_array_1[0, :3] = None
        numpy_array_1[0, 3:8] = arrayheight_tiptobase[0][92:]  # 11
        numpy_array_1[0, 8:] = None

        # Plot the heatmap.
        fig, ax = plt.subplots(figsize=(20, 10))
        im = plt.imshow(numpy_array_1, cmap='Reds', vmin=yaxis_min,
                        vmax=yaxis_max)

        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("height(units:um)", rotation=-90, va="bottom")

        # annotating heatmap by text
        annotate_heatmap(im, valfmt="{x:.1f}")

        return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

# 한 개만 보고 싶을 때 코드


data_Name = "./Mitutoyo/높이/QSEMO50025PPV1.xlsx"
makingheatmap(data_Name, 450, 550)


# Sample 여러개일 때 돌리는 코드

## fname = []
# data 중 가장 첫번째 번호
## sample_num1 = 1
# data 중 가장 마지막 번호
## sample_num2 = 3
# for i in range(sample_num1, sample_num2 + 1):
# fname.append(f"Data/NT7_InjMol_Monument800/20190724_data{i}.cs## v")

# dataname =##  []
##
# 총 측정할 샘플 갯수
# Num_sample ## = 3

# for makingdata in range(Num_sample):
# dataname.append(pd.read_csv(fname[makingdata],
# engine='python', index_col=Fals## e))

## makingheatmap(dataname, 720, 820, Num_sample=Num_sample)


# Subplot으로 히스토그램까지 다 그리고 싶을때.
#ax[1].hist(dnumpy, bins=20)
#im1 = ax1.hist(dnumpy, bins = 10, range = (720,820))

plt.show()
