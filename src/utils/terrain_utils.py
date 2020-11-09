import logging
import pandas as pd
import numpy as np
from podpac import Coordinates, clinspace
from podpac.datalib import TerrainTiles
import geopandas as gpd
import shapely.geometry as shp
from scipy.interpolate import UnivariateSpline
from shapely.geometry import MultiPolygon
import imcmc
logging.basicConfig(level=logging.INFO)


def interpolate_contour(points, n_pts=None, s = None,curve=False):
    if n_pts is None:
        n_pts = 100
    if s is None:
        s= points.shape[0] / 5
    # Linear length along the line:
    if not curve:
        points[-1, :] = points[0, :]
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]

    # Build a list of the spline function, one for each dimension:
    splines = [UnivariateSpline(distance, coords, k=2, s=s) for coords in points.T]

    # Computed the spline for the asked distances:
    alpha = np.linspace(0, 1.00, n_pts)
    points_fitted = np.vstack(spl(alpha) for spl in splines).T
    if not curve:
        points_fitted[-1, :] = (points_fitted[0, :] + points_fitted[-1, :] + points_fitted[-2, :] + points_fitted[1, :])*0.25

    return points_fitted

def create_offset_contour(contour, offset=10):
    # Create offset
    # Create a Polygon from the 2d array
    contour_poly = shp.Polygon(contour)
    poly_offset = contour_poly.buffer(offset)  # Outward offset
    # Turn polygon points into numpy arrays for plotting
    if type(poly_offset) != MultiPolygon:
        contour_offset = np.array(poly_offset.exterior)
    else:
        contour_offset = contour
    return contour_offset


def retrieve_data(zoom=10,
                  latlong_center=(51.062683, -115.401409),
                  dlat=0.03,
                  dlong: float = 0.03,
                  n_lat=2000,
                  n_long=2000,
                  in_crs='EPSG:4326'):
    node = TerrainTiles(tile_format='geotiff', zoom=zoom, cache_ctrl=['disk'],
                        cache_output=True)

    c = Coordinates([clinspace(latlong_center[0] - dlat,
                               latlong_center[0] + dlat, n_lat),
                     clinspace(latlong_center[1] - dlong,
                               latlong_center[1] + dlong, n_long)], dims=['lat', 'lon'])
    # evaluate node
    logging.warning('Getting DEM ...')
    o = node.eval(c)
    o.name = 'topo'
    df = o.to_dataframe().reset_index()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon,
                                                           df.lat),
                           crs='EPSG:4326').to_crs(in_crs)
    return o, df, gdf


def create_line_df(img, skip=3):
    lines = []
    for i in np.arange(0, img.shape[1], skip):



        line_df = pd.DataFrame({"x": np.ones(img.shape[1]) * i,
                                "y": np.arange(img.shape[1]),
                                "z": img[:, i],
                                "id": np.ones(img.shape[0]) * i})
        lines.append(line_df)
    lines = pd.concat(lines)

    return lines


def generate_perlin_noise_2d(shape, res, tileable=(False, False)):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[:-d[0], :-d[1]]
    g10 = gradients[d[0]:, :-d[1]]
    g01 = gradients[:-d[0], d[1]:]
    g11 = gradients[d[0]:, d[1]:]
    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5, lacunarity=2, tileable=(False, False)):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency * res[0], frequency * res[1]), tileable)
        frequency *= lacunarity
        amplitude *= persistence
    return noise
