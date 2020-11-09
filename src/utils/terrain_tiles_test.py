from podpac.datalib.terraintiles import TerrainTiles
from podpac import Coordinates, clinspace
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import geopandas as gpd
from sklearn.neighbors import KNeighborsClassifier
import logging
import cartopy.crs as ccrs

# Read in JSON with the parks:
@st.cache
def load_parks(crs='ABC'):
    parks_all = gpd.read_file('/home/anton/Repos/topo_data/national_parks_boundaries.shp')
    parks_all = parks_all.to_crs(crs)
    return parks_all


parkname = 'Banff National Park of Canada'
# create terrain tiles node

node = TerrainTiles(tile_format='geotiff', zoom=10, cache_ctrl=['disk'],
                    cache_output=True)
logging.warning('Loading parks...')
parks_all = load_parks(crs='EPSG:4326')
df_parks = parks_all[parks_all['parkname_e'] == parkname]

# latlong_center = 51.062683, -115.401409
# dlat = 0.04
# dlong = 0.04
#
# create coordinates to get tiles
latlong_center = [(df_parks.bounds.miny.iloc[0] + df_parks.bounds.maxy.iloc[0]) / 2, \
                  (df_parks.bounds.minx.iloc[0] + df_parks.bounds.maxx.iloc[0]) / 2]
dlat = df_parks.bounds.maxy.iloc[0] - latlong_center[0]
dlong = df_parks.bounds.maxx.iloc[0] - latlong_center[1]

n_lat = 2000
n_long = 2000
c = Coordinates([clinspace(latlong_center[0] - dlat,
                           latlong_center[0] + dlat, n_lat),
                 clinspace(latlong_center[1] - dlong,
                           latlong_center[1] + dlong, n_long)], dims=['lat', 'lon'])

# evaluate node
w, h = 10, 12
logging.warning('Getting DEM ...')
o = node.eval(c)
o.name = 'topo'
# Retrieve parks in CRS of the node

fig, ax = plt.subplots(dpi=1500)

df = o.to_dataframe().reset_index()

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon,
                                                       df.lat),
                       crs='EPSG:4326').to_crs('EPSG:4326')

p = df_parks.geometry.iloc[0]
# Apply on a subset, train a KNN, apply on all
logging.warning('Training inpoly...')

with st.spinner('Training InPoly...'):
    gdf_subset = gdf.sample(n=5000,random_state=123)
    gdf_subset['in_park'] = gdf_subset.geometry.apply(lambda x: x.within(p))
    clf = KNeighborsClassifier(n_neighbors=3, n_jobs=8)
    clf.fit(gdf_subset[['lat','lon']],gdf_subset['in_park'])
logging.warning('Predicting inpoly...')
with st.spinner('Predicting In Poly...'):
    gdf['in_park'] = clf.predict(gdf[['lat','lon']])


mask = gdf['in_park'].values.reshape(n_lat, n_long)
masked_data = np.ma.masked_array(o.data, mask=np.logical_not(mask))

# Transform coordinates:
new_coords = c.transform(crs='EPSG:3395').coords

logging.warning('Contouring...')
dist_height = 300
cont = ax.contour(new_coords['lon'], new_coords['lat'], masked_data, levels=np.arange(1000, 3500, dist_height),
                  colors='black', linewidths=0.1)
df_parks.to_crs('EPSG:3395').plot(ax=ax,color='white', edgecolor='black')

ax.set_xlim([new_coords['lon'].min(),new_coords['lon'].max()])
ax.set_ylim([new_coords['lat'].min(),new_coords['lat'].max()])

#ax.set_aspect(1.8)
# cl = ax.clabel(cont,inline=1,fmt = '%1.0f',levels = np.arange(1000,2850,200))
# o.plot()
# cont_tmp = ax_tmp.contour(o.lon, o.lat, masked_data, levels=np.arange(1000, 2850, 100),
#                   colors='black', linewidths=0.3)
logging.warning('Saving figure...')

fig.savefig(f'tmp_{dist_height}.png')
#st.write(fig)

#st.write(fig_tmp)
