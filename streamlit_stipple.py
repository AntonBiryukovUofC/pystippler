import os

import altair as alt
import imageio
import numpy as np
import pandas as pd
import streamlit as st
from scipy import ndimage as ndi
from scipy.ndimage import zoom
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

from src.rougier.stippler_utils import stipple, normalize


def quantify_color(img_orig, n_colors):
    # Load Image and transform to a 2D numpy array.
    img = img_orig[:, :, [0, 1, 2]] / 255
    w, h, d = original_shape = tuple(img.shape)
    assert d == 3
    image_array = np.reshape(img, (w * h, d))

    print("Fitting model on a small sub-sample of the data")
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    # Get labels for all points
    print("Predicting color indices on the full image (k-means)")
    labels = kmeans.predict(image_array)
    shape = (w, h, d)
    return labels, kmeans, shape


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


# Size the points
def pick_size(points, density, ps):
    Pi = points.astype(int)
    X = np.maximum(np.minimum(Pi[:, 0], density.shape[1] - 1), 0)
    Y = np.maximum(np.minimum(Pi[:, 1], density.shape[0] - 1), 0)
    sizes = (ps[0] + (ps[1] - ps[0]) * density[Y, X])
    return sizes


def pick_color(points, img_compressed, hex=True):
    from matplotlib.colors import rgb2hex
    Pi = points.astype(int)
    X = np.maximum(np.minimum(Pi[:, 0], img_compressed.shape[1] - 1), 0)
    Y = np.maximum(np.minimum(Pi[:, 1], img_compressed.shape[0] - 1), 0)
    cols = img_compressed[Y, X, :]
    if hex:
        hex_cols = np.array(cols.shape[0]*["#FFFFFF"])
        for i in range(cols.shape[0]):
            hex_cols[i] = rgb2hex(cols[i, :])
        res = hex_cols
    else:
        res = cols

    return res


@st.cache
def create_stipple(density, n_point, n_iter, filename, threshold):
    dirname = os.path.dirname(filename)
    basename = (os.path.basename(filename).split('.'))[0]
    pdf_filename = os.path.join(dirname, basename + "-stipple.pdf")
    png_filename = os.path.join(dirname, basename + "-stipple.png")
    dat_filename = os.path.join(dirname, basename + "-stipple.npy")
    density = 1.0 - normalize(density)
    density = density[::-1, :]

    density_new, points, bbox = stipple(n_point=n_point, n_iter=n_iter,
                                        density=density,
                                        force=True, dat_filename=dat_filename,
                                        png_filename=png_filename,
                                        pdf_filename=pdf_filename,
                                        pointsize=ps)
    return density_new, points, bbox


st.title('Stippling test')
filename = st.sidebar.text_input(label='File location', value='./data/plants.jpg')
n_point = st.sidebar.number_input(label=' N points', value=20000, max_value=50000, step=1000)
n_colors = st.sidebar.number_input(label=' N colors', value=3, max_value=15, min_value = 2)

n_iter = st.sidebar.number_input(label=' N iters', value=3, max_value=50, step=2)
threshold = st.sidebar.number_input(label=' Threshold', value=255., max_value=255., step=0.1, min_value=0.0)
ps_min = st.sidebar.number_input(label='Min point size ', value=1., min_value=0.01, max_value=100.0)
ps_max = st.sidebar.number_input(label='Max point size ', value=10., min_value=0.01, max_value=100.0)
ps = (ps_min, ps_max)

n_density = 200

density = imageio.imread(filename, pilmode='L')
img_original = imageio.imread(filename)

density_orig = density.copy()
# We want (approximately) 500 pixels per voronoi region
zoom_lvl = (n_point * n_density) / (density.shape[0] * density.shape[1])
zoom_lvl = int(round(np.sqrt(zoom_lvl)))
density = zoom(density, zoom_lvl, order=0)
density = np.minimum(density, threshold)

density_new, points, bbox = create_stipple(density, n_point, n_iter, filename, threshold)

# Do a visual
# st.write(points)
# Plot density:
st.write(density_new.shape)

density_thresh = np.minimum(density, threshold)
density_thresh = normalize(density_thresh)
# density_thresh = density_thresh[::-1, :]

points_df = pd.DataFrame(points, columns=['x', 'y'])


def preprocess_size(im):
    im = zoom(im, zoom_lvl, order=0)
    im = ndi.gaussian_filter(im, 1)
    new_im = 1 - normalize(im)

    return new_im


density_adjusted = preprocess_size(density_orig)

st.image(density_thresh, use_column_width=True)
# st.image(density_new, use_column_width=True)

# Quantify colors
labels, kmeans, shape = quantify_color(img_original,
                                       n_colors=n_colors)
img_compressed = recreate_image(kmeans.cluster_centers_, labels, shape[0], shape[1])
# Pick the size of the points using density as a guide:
points_df['size'] = pick_size(points, density_adjusted, ps)
# Pick the colors of the points using quantized space:
# points_df['color'] = pick_color(points, img_compressed, hex=True)
cols = pick_color(points, img_compressed, hex=True)
if n_colors > 2:
    points_df['color'] = cols
else:
    points_df['color'] = 'black'


width = 800
height = density.shape[0] / density.shape[1] * width
ch = alt.Chart(data=points_df,
               width=width,
               height=height).encode(x=alt.X('x', axis=None), y=alt.Y('y', axis=None, sort=alt.SortOrder('descending')),
                                     size=alt.Size('size', legend=None, scale=alt.Scale(range=(ps[0], ps[1]))),
                                     color=alt.Color('color',scale=None)).mark_point(filled=True, opacity=1.0)

ch = ch.configure_axis(grid=False).configure_view(strokeWidth=0)
st.altair_chart(ch)
st.image(img_compressed,use_column_width=True)
st.write(points_df['color'].value_counts())
#     # Apply threshold onto image
#     # Any color > threshold will be white
