import h5py
import numpy as np
import numpy.ma as ma
from scipy import spatial
import shapely
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import shapely.affinity
# from shapely.affinity import translate, scale

from PIL import Image, ImageDraw
from scipy.interpolate import griddata



class FieldGrid(object):
    def __init__(self, df, stride=1):
        self._df = df
        self._bounds = None
        self._outside = None
        self._mask = None
        self._stride = stride
        self._grids = {}


    @property
    def bounds(self):
        if self._bounds is None:
            xmin = min(self._df.Easting)
            xmax = max(self._df.Easting)
            ymin = min(self._df.Northing)
            ymax = max(self._df.Northing)
            self._bounds = (xmin, ymin, xmax, ymax)
        return self._bounds



    @property
    def outside(self):
        # print 'Generating outside boundary'
        if self._outside is None:
            self.boundary_polygon()

        # print 'Generated outside boundary'
        return self._outside

    @property
    def mask(self):
        if self._mask is None:
            self.grid_mask()

        return self._mask

    @property
    def stride(self):
        return self._stride

    def alpharadius(self, pa, pb, pc):
        # Lengths of sides of triangle
        # Have to take the square root to get actual length. We don't really
        # care too much, because exponentiation is monotonic for positive values,
        # thus it preserves max.

        def square(x):
            return x*x

        a = square(pa[0]-pb[0]) + square(pa[1]-pb[1])
        b = square(pb[0]-pc[0]) + square(pb[1]-pc[1])
        c = square(pc[0]-pa[0]) + square(pc[1]-pa[1])

        circum_r = max(a,b,c)
        return circum_r

    def boundary_polygon(self, alpha=1200):
        points = self._df[['Easting','Northing']].values
        # print 'Calculating Tesselation'
        tri = spatial.Delaunay(points,qhull_options='Qbb Qt Fv')
        # print 'Done calculating Tesselation.'

        count = 0


        polys = []
        alphas = []
        for ia, ib, ic in tri.vertices:
            pa = points[ia]
            pb = points[ib]
            pc = points[ic]
            circum_r = self.alpharadius(pa,pb,pc)
            # alphas.append(circum_r)
            # Here's the radius filter.
            if circum_r < alpha:
                count += 1
                pts = [points[idx] for idx in [ia,ib,ic,ia]]
                poly = shapely.geometry.Polygon(pts)

                if poly.is_valid:
                    polys.append(poly)
                else:
                    print "Invalid polygon"

        outside = cascaded_union(polys)
        # Dilate to incorporate header width
        # Width is in inches, convert to meters.
        outside = outside.buffer(self._df['Width'].max() / 39.37)

        # print 'Finished calculating boundary'
        self._outside = outside
        return self._outside

    def grid_mask(self, origin=None):
        outside = self.outside
        scale = float(1) / self.stride

        (xmin,ymin,xmax,ymax) = self.bounds
        # print 'Making mask with bounds %s' % (self.bounds,)
        xtrans = -xmin
        ytrans = -ymin



        # print 'Translating (%s, %s)' % (xtrans, ytrans)

        outside = shapely.affinity.translate(outside, xtrans, ytrans)
        outside = shapely.affinity.scale(outside, scale, scale, origin=(0, 0))

        # (xmin,ymin,xmax,ymax) = bounds.bounds


        width = int((xmax - xmin) * scale) + 1
        height = int((ymax - ymin) * scale) + 1
        img = Image.new('L', (width, height), 0)

        if outside.geom_type == 'Polygon':
            polys = [outside]
        else:
            polys = outside.geoms

        for poly in polys:
            ImageDraw.Draw(img).polygon(poly.exterior.coords, outline=0, fill=1)
            for inner in poly.interiors:
                ImageDraw.Draw(img).polygon(inner.coords, outline=0, fill=0)

        self._mask = np.flipud(np.array(img))
        return self._mask


    def grid_data(self, obs='Yield'):

        stride = self._stride
        samples = self._df

        # Then turn that into a mask
        mask = self.mask # bounds_mask(outside, scale=(1/stride,1/stride))
        mask = mask == 0.0
        # print 'Mask shape'
        # print mask.shape
        # plt.figure()
        # plt.imshow(mask)

        # Create the grid coordinates
        (oxmin,oymin,oxmax,oymax) = self.bounds
        x = np.arange(oxmin,oxmax,stride)
        y = np.arange(oymin,oymax,stride)
        # xx,yy = np.meshgrid(x,y,indexing='ij')
        xx,yy = np.meshgrid(x,y)
        xx = ma.masked_array(xx,mask)
        yy = ma.masked_array(yy,mask)
        # print 'Grid Shape'
        # print xx.shape

        # Interpolate the data
        x = samples['Easting']
        y = samples['Northing']
        z = samples[obs]

        # print 'Samples min %s' % min(z)
        # print 'Samples max %s' % max(z)

        grid = griddata((x,y),z,(xx,yy),method='linear')
        # Flip the grid to match the orientation of the mask
        grid = np.flipud(grid)

        # grid = ma.masked_invalid(ma.masked_array(grid, mask))
        # grid = ma.masked_invalid(ma.masked_less(ma.masked_array(grid, mask), 0))
        grid = ma.masked_invalid(ma.masked_array(grid, mask))

        return grid

    def groupby(self, groups, reset_bounds=False):
        return FieldGridGroupBy(groups, self, reset_bounds=False)

    def __getattr__(self, attr):
        if not self._grids.has_key(attr):
            grid = self.grid_data(attr)
            self._grids[attr] = grid
        return self._grids[attr]



class FieldGridGroupBy():

    def __init__(self, groups, fg, reset_bounds=False):
        self.cache = {}
        for (g,df2) in fg._df.groupby(groups):
            fg2 = FieldGrid(df2)
            fg2._stride = fg.stride
            if not reset_bounds:
                # print 'Preserving Bounds %s' %  (fg.bounds,)
                fg2._bounds = fg.bounds
            self.cache[g] = fg2

    def __iter__(self):
        return self.cache.iteritems()

    def __len__(self):
        return len(self.cache)




def generate_hd5(input_dataset, filename='/tmp/bitstead.hdf5'):
    with h5py.File(filename, "w") as f:
        for fname, dataset in input_dataset.groupby('Field'):
            print 'Processing field %s' % fname
            gsource = FieldGrid(dataset)
            for info, data in gsource.groupby(['Season', 'Commodity']):
                year, commodity = info
                print 'Processing info for %s %s' % (year, commodity)
                try:
                    grp = f.create_group('%s/%s/%s/yield' % (fname, year, commodity))
                except:
                    grp = f['%s/%s/%s/yield' % (fname, year, commodity)]

                grp.create_dataset('mask', data=data.mask, compression='gzip', compression_opts=4)
                grp.create_dataset('yield', data=data.Yield, compression='gzip', compression_opts=4)
                grp.create_dataset('moisture', data=data.Moisture_s, compression='gzip', compression_opts=4)
                grp.create_dataset('elevation', data=data.Elevation, compression='gzip', compression_opts=4)

                xmin, ymin, xmax, ymax = data.bounds
                stride = data._stride

                grp.attrs['xmin'] = xmin
                grp.attrs['ymin'] = ymin
                grp.attrs['xmax'] = xmax
                grp.attrs['ymax'] = ymax
                grp.attrs['stride'] = stride
