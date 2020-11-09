import imageio
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from skimage.filters import sobel, sobel_v, sobel_h
from skimage import feature



filename = '/home/anton/Documents/skully01.png'
img = imageio.imread(filename, pilmode='L').astype(np.float64)
fig, ax = plt.subplots(figsize=(15, 15))
ny, nx = img.shape
gd = 25

resampled_img = img[::gd, ::gd]  # sampling

ax.imshow(img)

# axs.plot(probs)


st.write("Extrude depth map")
lines = []
for i in np.arange(0, img.shape[0]):
    line_df = pd.DataFrame({"y": np.ones(img.shape[1]) * i,
                            "x": np.arange(img.shape[1]),
                            "z": img[img.shape[0]-i-1, :],
                            "id": np.ones(img.shape[1]) * i})

    lines.append(line_df)

lines = pd.concat(lines)
# lines.loc[lines['z'] > 254,'z'] = 0
lines = lines.loc[lines['z'] < 254]
lines['size'] = 15

np.random.seed(10)
ids_unique = lines['id'].unique().astype(np.int64)
#lines_plot = lines[lines['id'].isin(id_random)].sort_values(['y', 'x']).set_index('id', drop=False)
lines_plot = lines.sort_values(['y', 'x']).set_index('id', drop=False)
# Create plotly figure using low-lvl interface:
fig = go.Figure(layout = dict(paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)'))
st.write(lines_plot.sample(20))
for i in lines_plot['id'].unique()[::-1]:
    one_line_df = lines_plot.loc[i]
    one_line_df['z_new'] =one_line_df['z']/1
    trace = go.Scatter(x=one_line_df['x'],
                         y=one_line_df['y'] + one_line_df['z_new'],
                         showlegend=False,
                         mode='lines',
                       marker={'size': 0.1},
                       line={'width': 0.3},
                       marker_color='rgba(0, 0, 0, .99)',
                       fill='tozeroy',
                       fillcolor='white'
                       )

    if np.random.uniform(0, 1) > 0.5:
        fig.add_trace(trace)
camera = dict(
    eye=dict(x=0, y=0., z=-2),
    up=dict(x=0, y=-1, z=0)
)
fig.update_layout(height=800,
                width=800,
                  scene_xaxis_showbackground=False,
                  scene_yaxis_showbackground=False,
                  scene_zaxis_showbackground=False,

                  )
st.plotly_chart(fig)
