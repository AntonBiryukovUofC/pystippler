from itertools import product
from tqdm.contrib.concurrent import thread_map
import pandas as pd
import requests
from podpac import Coordinates, clinspace


def get_altitude(lat_lon):
    r = requests.get(url=cdsm_url, params={'lat': lat_lon[0],
                                           'lon': lat_lon[1]})
    res = r.json()
    alt = res['altitude']
    lat = res['geometry']['coordinates'][1]
    lon = res['geometry']['coordinates'][0]
    tmp = {'alt':alt,'lat':lat,'lon':lon}

    return tmp


cdsm_url = 'http://geogratis.gc.ca/services/elevation/cdsm/altitude'
latlong_center = 50.437865, -116.218796
dlat = 0.017
dlong = 0.0175
n_long = 1000
n_lat = 1000

c = Coordinates([clinspace(latlong_center[0] - dlat,
                           latlong_center[0] + dlat, n_lat),
                 clinspace(latlong_center[1] - dlong,
                           latlong_center[1] + dlong, n_long)], dims=['lat', 'lon']).coords
lat_long = list(product(c['lat'], c['lon']))

r = requests.get(url=cdsm_url, params={'lat': lat_long[0][0],
                                       'lon': lat_long[0][1]})
results = thread_map(get_altitude, lat_long[:20000], max_workers=100)
print(results)
