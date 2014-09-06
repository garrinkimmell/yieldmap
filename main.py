import yieldmap.ag_leader_advanced as ala
import yieldmap.gridding as grid
import yieldmap.mapping as mapping
from pyproj import Proj

import sys


if __name__ == '__main__':
    inputfile = sys.argv[1]
    # Read in the raw data file
    # Do the necessary preprocessing
    data = ala.read_file(inputfile)

    # Project the samples.
    kansas = Proj(init='epsg:2796')
    east, north = kansas(data.Longitude.values, data.Latitude.values)
    data['Easting'] = east
    data['Northing'] = north

    # Create the grid file

    grid.generate_hd5(data, filename='/tmp/test.hdf5')
    # mapping.generate_plots(fname='/tmp/test.hdf5')

    print data.head()
    print 'main'
