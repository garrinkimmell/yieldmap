import numpy as np
import pandas as pd

types = {'Commodity': object,
         'Distance': np.float32,
         'Elevation': np.float32,
         'Field': object,
         'Heading': np.float32,
         'Latitude': np.float64,
         'Longitude': np.float64,
         'Load': object,
         'Mappable': np.int32,
         'Season': np.int32,
         'UTC': np.int32,
         'Width': np.float32,
         'Duration': np.float32,
         'Moisture_s': np.float32,
         'Mass_Flow_': np.float32,
         # Area is calculated
         'Area': np.float32,
         # Yield is calculated
         'Yield': np.float32}

# In the Agleader file generated, field names are represented as `FXX: Foo`,
converters = {'Field': lambda n: n.split(':')[-1].strip().upper()}


def read_file(fname):
    cnts = pd.read_csv(fname, dtype=types, usecols=types.keys(), converters=converters)
    return cnts
