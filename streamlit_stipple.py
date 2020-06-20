import os

import altair as alt
import imageio
import numpy as np
import pandas as pd
import streamlit as st
from scipy import ndimage as ndi
from scipy.ndimage import zoom
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.utils import shuffle
from skimage import color
from matplotlib.colors import to_rgb
from src.rougier.stippler_utils import stipple, normalize
from skimage.color import deltaE_ciede2000, deltaE_cie76

NDENSITY = 200
# black, brown, blue, green, orange, sky blue, red, red orange, violet, white, yellow, yellow green
PALETTE = ('#49484C', '#6F5756', '#3D57D5', '#508D5D', '#E16B40',
           '#64B1F2', '#CE465D', '#D95555', '#605CA6','#FFFFFF','#FDF453', '#87BE55')

@st.cache
def to_discrete_colors(img, palette=PALETTE):
    # Convert to LAB
    img_lab = color.rgb2lab(img)
    h, w = img.shape[0], img.shape[1]
    nn = KNeighborsClassifier(n_neighbors=1,
                              n_jobs=8)
    color_palette_train = np.expand_dims(np.array([to_rgb(c) for c in palette]).reshape(-1, 3), 1)

    color_palette_train = color.rgb2lab(color_palette_train)
    y = np.arange(color_palette_train.shape[0])
    # Fit predict in lab space
    nn.fit(color_palette_train[:, 0, :], y)
    img_predicted = nn.predict(img_lab.reshape(-1, 3))
    img_predicted = color_palette_train[img_predicted.reshape(h, w)][:, :, 0, :]
    img_predicted = color.lab2rgb(img_predicted)
    return img_predicted


def preprocess_size(im, zoom_lvl):
    im = zoom(im, zoom_lvl, order=0)
    im = ndi.gaussian_filter(im, 1)
    new_im = 1 - normalize(im)

    return new_im


def quantify_color(img_orig, n_colors):
    # Load Image and transform to a 2D numpy array.
    img = img_orig[:, :, [0, 1, 2]] / 255

    w, h, d = tuple(img.shape)
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
        hex_cols = np.array(cols.shape[0] * ["#FFFFFF"])
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
                                        pointsize=ps, n_density=NDENSITY)
    return density_new, points, bbox


st.title('Stippling test')
filename = st.sidebar.text_input(label='File location', value='./data/rainbow-emoji.jpg')
n_point = st.sidebar.number_input(label=' N points', value=5000, max_value=50000, step=1000)
n_colors = st.sidebar.number_input(label=' N colors', value=3, max_value=15, min_value=2)
quantify_color_flag = st.sidebar.checkbox(label='Quantify colors?', value=False)
n_iter = st.sidebar.number_input(label=' N iters', value=3, max_value=50, step=2)
threshold = st.sidebar.number_input(label=' Threshold', value=255., max_value=255., step=0.1, min_value=0.0)
ps_min = st.sidebar.number_input(label='Min point size ', value=1., min_value=0.01, max_value=100.0)
ps_max = st.sidebar.number_input(label='Max point size ', value=10., min_value=0.01, max_value=100.0)
ps = (ps_min, ps_max)

# Read the images in B/W and Full Color
density = imageio.imread(filename, pilmode='L')
img_original = imageio.imread(filename)

img_discrete_palette = to_discrete_colors(img_original, palette=PALETTE)

density_orig = density.copy()
# We want (approximately) NDENSITY pixels per voronoi region
zoom_lvl = (n_point * NDENSITY) / (density.shape[0] * density.shape[1])
zoom_lvl = int(round(np.sqrt(zoom_lvl)))
density = zoom(density, zoom_lvl, order=0)
density = np.minimum(density, threshold)

density_new, points, bbox = create_stipple(density, n_point, n_iter, filename, threshold)

density_thresh = np.minimum(density, threshold)
density_thresh = normalize(density_thresh)
# Convert to DF for altair viz
points_df = pd.DataFrame(points, columns=['x', 'y'])
density_adjusted = preprocess_size(density_orig, zoom_lvl=zoom_lvl)
points_df['size'] = pick_size(points, density_adjusted, ps)
st.markdown(f"###  Thresholded at {threshold} B/W source image")
st.image(density_thresh, use_column_width=True)

# Quantify colors
if (len(img_original.shape) > 2) & quantify_color_flag:
    img_discrete_palette = to_discrete_colors(img_original, palette=PALETTE)
    img_compressed = img_discrete_palette
    st.markdown(f'### Image discretized into color palette \n{PALETTE}')
    st.image(img_compressed, use_column_width=True)
    cols = pick_color(points, img_compressed, hex=True)
    points_df['color'] = cols
else:
    points_df['color'] = 'black'

# Pick the size of the points using density as a guide:


width = 800
height = density.shape[0] / density.shape[1] * width
ch = alt.Chart(data=points_df,
               width=width,
               height=height).encode(x=alt.X('x', axis=None), y=alt.Y('y', axis=None, sort=alt.SortOrder('descending')),
                                     size=alt.Size('size', legend=None, scale=alt.Scale(range=(ps[0], ps[1]))),
                                     color=alt.Color('color', scale=None)).mark_point(filled=True, opacity=1.0)
ch = ch.configure_axis(grid=False).configure_view(strokeWidth=0)
st.markdown('### Stippled image')
st.altair_chart(ch)
st.write(points_df['color'].value_counts())
#     # Apply threshold onto image
#     # Any color > threshold will be white
