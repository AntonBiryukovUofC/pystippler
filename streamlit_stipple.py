import io
import os

import altair as alt
import imageio
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from matplotlib.colors import rgb2hex
from matplotlib.colors import to_rgb
from scipy import ndimage as ndi
from scipy.ndimage import zoom
from skimage.transform import rescale
from sklearn.neighbors import KNeighborsClassifier

from src.rougier.stippler_utils import stipple, normalize, PALETTE, _check_size

NDENSITY = 200
st.set_page_config(layout="wide")
alt.renderers.set_embed_options(actions=False)


# We really need to see training messages only when cache is new
def show_palette(palette):
    palette_df = pd.DataFrame({'color': palette, 'order': np.arange(len(palette)),
                               'y': np.ones(len(palette))})

    ch = alt.Chart(data=palette_df,
                   height=35, width=400).encode(x=alt.X('order:O', axis=None),
                                                y=alt.Y('y', axis=None),
                                                fill=alt.Fill('color:N',
                                                              scale=alt.Scale(
                                                                  domain=palette,
                                                                  range=palette), legend=None),
                                                color=alt.value('black')
                                                ).mark_rect(strokeWidth=1, stroke='black')
    ch = ch.configure_axis(grid=False).configure_view(strokeWidth=0)
    return ch


@st.cache(suppress_st_warning=True)
def to_discrete_colors(img, palette=PALETTE, zoom_lvl=2, zoom_shrink=4):
    img_lab = img
    # Speed up predict by shrinking the image
    img_lab_mini = rescale(img_lab, 1 / zoom_shrink, multichannel=True)
    h, w = img_lab_mini.shape[0], img_lab_mini.shape[1]
    # Populate NN space with the palette colors
    nn = KNeighborsClassifier(n_neighbors=1,
                              n_jobs=8)
    color_palette_train = np.expand_dims(np.array([to_rgb(c) for c in palette]).reshape(-1, 3), 1)
    y = np.arange(color_palette_train.shape[0])
    # Fit predict in lab space

    nn.fit(color_palette_train[:, 0, :], y)

    print(img_lab_mini.shape)
    img_predicted = nn.predict(img_lab_mini.reshape(-1, 3)).reshape(h, w)
    # Zoom back
    img_predicted = zoom(img_predicted, zoom_shrink * zoom_lvl, order=0)

    img_predicted = color_palette_train[img_predicted][:, :, 0, :]
    img_predicted = img_predicted

    return img_predicted


def preprocess_size(im, zoom_lvl):
    """
    Resize and normalize input image for stippling
    :param im:
    :param zoom_lvl:
    :return:
    """
    im = zoom(im, zoom_lvl, order=0)
    im = ndi.gaussian_filter(im, 1)
    new_im = 1 - normalize(im)

    return new_im


def pick_size(points, density, ps):
    """
    Resize the points based on density of the image to interval  [ps[0],ps[1]]
    :param points:
    :param density:
    :param ps:
    :return:
    """
    Pi = points.astype(int)
    X = np.maximum(np.minimum(Pi[:, 0], density.shape[1] - 1), 0)
    Y = np.maximum(np.minimum(Pi[:, 1], density.shape[0] - 1), 0)
    sizes = (ps[0] + (ps[1] - ps[0]) * normalize(density[Y, X]))
    return sizes


def pick_color(points, img_compressed, hex=True):
    """ Pick color from an underlying image"""
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


@st.cache(allow_output_mutation=True)
def create_stipple(density, n_point, n_iter, filename):
    dirname = os.path.dirname(filename)
    basename = (os.path.basename(filename).split('.'))[0]
    dat_filename = os.path.join(dirname, basename + "-stipple.npy")
    density = 1.0 - normalize(density)
    density = density[::-1, :]

    density_new, points, bbox = stipple(n_point=n_point, n_iter=n_iter,
                                        density=density,
                                        force=True, dat_filename=dat_filename)
    return density_new, points, bbox


st.title('Create-a-Stipple')
st.markdown(
    'My attempt to set up a stippling procedure in Python. Inspired (and built on top of) by https://github.com/ReScience-Archives/Rougier-2017')
st.sidebar.title("Stippling parameters")
# Parameters section
# If local file is desired, uncomment below
# filename = st.sidebar.text_input(label='File location', value='./data/leaf.jpg')
uploaded_file = st.sidebar.file_uploader("Choose a image file", type=["png", 'jpg', 'jpeg', 'tiff'])
n_point = st.sidebar.number_input(label=' N points', value=5000, max_value=50000, step=1000)
quantify_color_flag = st.sidebar.checkbox(label='Quantify colors?', value=True)
n_iter = st.sidebar.number_input(label=' N iters', value=3, max_value=50, step=2)
threshold = st.sidebar.number_input(label=' Threshold', value=255., max_value=255., step=0.1, min_value=0.0)
ps_min = st.sidebar.number_input(label='Min point size ', value=1., min_value=0.01, max_value=100.0)
ps_max = st.sidebar.number_input(label='Max point size ', value=10., min_value=0.01, max_value=100.0)
jitter_x = st.sidebar.number_input(label='Jitter X', value=0.2, min_value=0.01, max_value=20.0)
jitter_y = st.sidebar.number_input(label='Jitter Y', value=0.2, min_value=0.01, max_value=20.0)
power = st.sidebar.number_input(label='Size power', value=1, min_value=1, max_value=20)
ps = (ps_min, ps_max)


if quantify_color_flag:
    st.header('Color Quantization panel')
    st.markdown('Colors provided by the Staedtler Broadliner pen set')
    palette_ch = show_palette(PALETTE)
    st.altair_chart(palette_ch)
bytes_img = None
# Placeholder for the final result
info_read = st.empty()

final_result_header = st.empty()
# Organize visuals in 3 columns: Original Img | B/W or quantified | Stipple:
col1, col2, col3 = st.beta_columns(3)
# TODO populate name here with the session ID
if uploaded_file is not None:
    # If first time in this session...
    bytes_img = io.BytesIO(uploaded_file.read())
    filename = uploaded_file.name
    info_read.info(f'Reading in {filename}..')
    if not(os.path.exists(filename)):
        density = np.array(Image.open(bytes_img).convert('L'))
        img_original = np.array(Image.open(bytes_img).convert('RGB'))
        # Check the size of a file here, and if too big - resize.
        density = _check_size(density,max_dim = 1000)
        img_original = _check_size(img_original, max_dim=1000)
        imageio.imsave(filename,img_original)
    else:
        # Already have this file, so let's read the cache
        density = np.array(Image.open(filename).convert('L'))
        img_original = np.array(Image.open(filename).convert('RGB'))
else:
    st.stop()
# Read the images in B/W and Full Color
# if working with local files, uncomment:
# density = imageio.imread(filename, pilmode='L')
# img_original = imageio.imread(filename, pilmode='RGB')

density_orig = density.copy()

# Pick a zoom level
# We want (approximately) NDENSITY pixels per voronoi region
zoom_lvl = (n_point * NDENSITY) / (density.shape[0] * density.shape[1])
zoom_lvl = max(int(round(np.sqrt(zoom_lvl))), 1)
density = zoom(density, zoom_lvl, order=0)
# Clip the density based on threshold, store normalized density separately for viz
density = np.minimum(density, threshold)
density_normalized = normalize(density)

density_new, points, bbox = create_stipple(density, n_point, n_iter, filename)

width = 400
height = density.shape[0] / density.shape[1] * width


# Convert to DF for altair viz
points_df = pd.DataFrame(points, columns=['x', 'y'])
density_adjusted = preprocess_size(density_orig, zoom_lvl=zoom_lvl)
points_df['size'] = pick_size(points, np.power(density_adjusted, power), ps)

col3.image(np.array(img_original),
           caption=f"Original",width = width,height = height)
# Quantify colors
if (len(img_original.shape) > 2) & quantify_color_flag:
    img_discrete_palette = to_discrete_colors(img_original, palette=PALETTE, zoom_lvl=zoom_lvl)

    col2.image(img_discrete_palette, caption='Quantized colors',width = width,height = height)
    cols = pick_color(points, img_discrete_palette, hex=True)
    points_df['color'] = cols
else:
    points_df['color'] = 'black'
    col2.image(density_normalized,
               caption=f"Thresholded at {threshold} B/W source image", width = width,height = height)

# Pick the size of the points using density as a guide:



points_df['x_jit'] = points_df['x'] + np.random.normal(0, jitter_x, points_df.shape[0])
points_df['y_jit'] = points_df['y'] + np.random.normal(0, jitter_y, points_df.shape[0])

ch = alt.Chart(data=points_df,
               width=width,
               height=height).encode(x=alt.X('x_jit', axis=None),
                                     y=alt.Y('y_jit', axis=None, sort=alt.SortOrder('descending')),
                                     size=alt.Size('size', legend=None, scale=alt.Scale(range=(ps[0], ps[1]))),
                                     fill=alt.Fill('color', scale=None),
                                     color=alt.value('black')).mark_point(opacity=1.0, strokeWidth=0.2)
ch = ch.configure_axis(grid=False).configure_view(strokeWidth=0)
final_result_header.header('Stippled image vs original')
col1.altair_chart(ch, )

