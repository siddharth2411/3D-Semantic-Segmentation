
import pandas as pd
import numpy as np
import pptk

def read_points(f):
    # reads Semantic3D .txt file f into a pandas dataframe
    col_names = ['x', 'y', 'z', 'i', 'r', 'g', 'b']
    col_dtype = {'x': np.float32, 'y': np.float32, 'z': np.float32, 'i': np.int32,
                  'r': np.uint8, 'g': np.uint8, 'b': np.uint8}
    return pd.read_csv(f, names=col_names, dtype=col_dtype, delim_whitespace=True)


def read_labels(f):
    # reads Semantic3D .labels file f into a pandas dataframe
    return pd.read_csv(f, header=None)[0].values



points = read_points('untermaederbrunnen_station3_xyz_intensity_rgb.txt')

points


labels = read_labels('untermaederbrunnen_station3_xyz_intensity_rgb.labels')

labels

v = pptk.viewer(points[['x', 'y', 'z']])
v.attributes(points[['r', 'g', 'b']] / 255., points['i'])
v.set(point_size=0.001)

mask = labels != 0

P = points[mask]
L = labels[mask]

v = pptk.viewer(P[['x', 'y', 'z']])
v.attributes(P[['r', 'g', 'b']] / 255., P['i'], L)
v.set(point_size=0.001)



