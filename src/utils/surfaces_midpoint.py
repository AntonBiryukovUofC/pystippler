import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import altair as alt
from sklearn.neighbors import KNeighborsClassifier, KernelDensity

surface_hw = pd.read_csv('/home/anton/Downloads/HW.csv')
surface_fw = pd.read_csv('/home/anton/Downloads/FW.csv')
# Bring scales to something easier to read
surface_fw['x'] -= surface_fw['x'].min()
surface_fw['y'] -= surface_fw['y'].min()
surface_hw['x'] -= surface_hw['x'].min()
surface_hw['y'] -= surface_hw['y'].min()

surface_hw['id'] = 'hw'
surface_fw['id'] = 'fw'

surface_df = pd.concat([surface_fw, surface_hw], axis=0)
surface_df['id_class'] = (surface_df['id'] == 'hw').astype(np.int64)
print(surface_df['id_class'].mean())
fig = px.scatter_3d(surface_df, x='x', y='y', z='z', color='id', opacity=0.2)
# 3D view
st.write('3D view after min correction')
st.plotly_chart(fig)

# Find a mid-surface:
# Sampler
kde = KernelDensity(bandwidth=20)
kde.fit(surface_df[['x', 'y', 'z']])
new_pts = kde.sample(n_samples=surface_df.shape[0] * 50, random_state=123)
new_pts_df = pd.DataFrame(new_pts, columns=['x', 'y', 'z'])

# Classifier
knn = KNeighborsClassifier(n_neighbors=10, weights='distance')
knn.fit(surface_df[['x', 'y', 'z']], surface_df['id_class'])
with st.spinner('Predicting...'):
    new_pts_class = knn.predict(new_pts_df)
new_pts_df['id_class'] = np.where(new_pts_class == 1, 'hw', 'fw')
print(new_pts.shape)
# 2D view
slice_y = surface_df['y'].mean()
slice_delta = 5

st.write(f'Side view at y={slice_y} , delta = {slice_delta}')
slice_df = surface_df[surface_df['y'].between(slice_y - slice_delta, slice_y + slice_delta)]

new_pts_slice_df = new_pts_df[new_pts_df['y'].between(slice_y - slice_delta, slice_y + slice_delta)]

ch = alt.Chart(slice_df, width= 800, height=600).encode(x='x', y='z', color='id').mark_line()
ch_new_pts = alt.Chart(new_pts_slice_df, width=800, height=600).encode(x='x', y='z',
                                                                       color='id_class').mark_point(opacity=0.15)

st.altair_chart(ch_new_pts + ch)
