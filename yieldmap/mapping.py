import h5py
import numpy.ma as ma
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_grid_map(data, **kw):
    y_formatter = mpl.ticker.ScalarFormatter(useOffset=False)
    #plt.gca().yaxis.set_major_formatter(y_formatter)
    plt.gca().axis('off')
    plt.imshow(data,**kw)

def get_inputs(field, year, commodity, areas):
    # Input costs
    # http://www.extension.iastate.edu/agdm/crops/pdf/a1-20.pdf

    seed_cost_unit = 3.78 # $/1000 Seeds
    seed_rate = 25 # 1000 Seeds/Acre

    n_cost_unit = 0.44 # Lbs

    if commodity == 'SOYBEANS':
        n_rate = 0 # Lbs/ACre
    else:
        n_rate = 100 # Lbs/Acre

    p_cost_unit = 0.43 # Lbs
    p_rate = 25 # Lbs/Acre

    k_cost_unit = 0.41 # Lbs
    k_rate = 25 # Lbs/Acre

    lime_cost_unit = 10 # $/Unit
    lime_rate =  1 # Unit/Acre

    herbicide_cost_unit = 26 # Dollars / Unit
    herbicide_rate = 1 # Units/Acre

    crop_insurance_cost_unit = 18 # Dollars / Unit
    crop_insurance_rate = 1 # Units/Acre

    machinery_cost_unit = 50 # Dollars / Unit
    machinery_rate = 1 # Unit/Acre

    input_costs = \
        seed_cost_unit * seed_rate +\
        n_cost_unit * n_rate + \
        p_cost_unit * p_rate + \
        k_cost_unit * k_rate + \
        lime_cost_unit * lime_rate + \
        herbicide_cost_unit * herbicide_rate + \
        crop_insurance_cost_unit * crop_insurance_rate + \
        machinery_cost_unit * machinery_rate

    print 'Per-unit input costs: $%s' % input_costs
    return input_costs

def get_market_price(field, year, commodity):
    if commodity == 'SOYBEANS':
        return 10
    else:
        return 5


def generate_plots(fname="/tmp/bitstead.hdf5", figbase='/tmp/figures'):
    with h5py.File("/tmp/bitstead.hdf5") as f:
        for fieldname,grp in f.iteritems():
            years = grp.keys()
            years.sort()
            print fieldname


            ylds = []
            profits = []

            for year in years:
                print '\t%s' % year
                for commodity in grp[year]:
                    print '\t\t%s' % commodity

                    data = grp[year][commodity]['yield']
                    xmin = data.attrs['xmin']
                    ymin = data.attrs['ymin']
                    xmax = data.attrs['xmax']
                    ymax = data.attrs['ymax']
                    stride = data.attrs['stride']
                    map_extents = [xmin,xmax,ymin,ymax]

                    # Map data
                    mask = data['mask'].value == 0
                    yld = ma.masked_less(ma.masked_invalid(ma.masked_array(data['yield'], mask)), 0)
                    moisture = ma.masked_array(data['moisture'], mask)
                    elevation = ma.masked_array(data['elevation'], mask)


                    # Areas
                    grid_cell_area = (stride ** 2) / 4046.86
                    areas = ma.masked_array(np.ones(mask.shape), mask) * grid_cell_area  # Convert square meters to acres


                    # Calculating profit/loss
                    costs = get_inputs(fieldname, year, commodity, areas) * areas

                    income = yld * areas * get_market_price(fieldname, year, commodity)
                    profit = income - costs
                    ppa = profit/areas

                    # Normalized yield
                    ynorm = (yld - ma.mean(yld)) / ma.std(yld)


                    # For summary info
                    ylds.append(ynorm)
                    profits.append(ppa)


                    try:

                        # Mask (for outline of field)
                        # fig = plt.figure()
                        # plt.ticklabel_format(useOffset=False, axis='y')
                        # plt.ticklabel_format(useOffset=False, axis='x')

                        # plt.imshow(mask, extent=map_extents)
                        # fig.savefig('%s/%s-%s-%s-mask.pdf' % (figbase, fieldname, year, commodity) , format='pdf')


                        # Yield Map
                        fig = plt.figure()
                        plt.title('%s %s Yield (Bu/Acre)' % (year, commodity) )
                        cmap = mpl.cm.get_cmap('RdYlGn')

                        norm = mpl.colors.BoundaryNorm(np.linspace(ma.min(yld),ma.max(yld),10), cmap.N)
                        plt.gca().axis('off')
                        plot_grid_map(yld, cmap=cmap, norm=norm, interpolation='none')
                        plt.colorbar()
                        fig.savefig('%s/%s-%s-%s-yield-map.pdf' % (figbase, fieldname, year, commodity) , format='pdf')

                        # Yield Histogram
                        fig = plt.figure()
                        plt.title('%s %s Yield Distribution' % (year, commodity) )
                        a = np.ravel(ma.compressed(yld))
                        Y,X = np.histogram(a, 10, normed=0, weights=(np.ones(a.shape) * grid_cell_area))
                        x_span = X.max()-X.min()
                        C = [cmap(((x-X.min())/x_span)) for x in X]
                        plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
                        plt.xlabel('Yield (Bushel/Acre)')
                        plt.ylabel('Acres')
                        fig.savefig('%s/%s-%s-%s-yield-histogram.pdf' %(figbase, fieldname, year, commodity), format='pdf')

                        # Normalized Yield Map
                        fig = plt.figure()
                        plt.title('%s %s Yield (Bu/Acre) Normalized' % (year, commodity) )
                        cmap = mpl.cm.get_cmap('RdYlGn')
                        norm = mpl.colors.BoundaryNorm(np.linspace(-5,5,11), cmap.N)
                        plot_grid_map(ynorm, cmap=cmap, norm=norm, interpolation='none')
                        plt.colorbar()
                        fig.savefig('%s/%s-%s-%s-yield-map-normalized.pdf' % (figbase, fieldname, year, commodity) , format='pdf')


                        # Input Costs Map
                        # fig = plt.figure()
                        # plt.title('%s %s Costs ($/Acre)' % (year, commodity) )
                        # cmap = mpl.cm.get_cmap('RdYlGn')
                        # norm = mpl.colors.BoundaryNorm(np.linspace(ma.min(costs/areas),ma.max(costs/areas),10), cmap.N)
                        # plot_grid_map(costs/areas, cmap=cmap, norm=norm, interpolation='none')
                        # plt.colorbar()
                        # fig.savefig('%s/%s-%s-%s-costs-map.pdf' % (figbase, fieldname, year, commodity) , format='pdf')

                        # Profit/Loss Map
                        fig = plt.figure()
                        plt.title('%s %s Profit/Loss ($/Acre)' % (year, commodity) )
                        cmap = mpl.cm.get_cmap('RdYlGn')
                        norm = mpl.colors.BoundaryNorm(np.linspace(ma.min(ppa),ma.max(ppa),10), cmap.N)
                        plot_grid_map(ppa, cmap=cmap, norm=norm, interpolation='none')
                        plt.colorbar()
                        fig.savefig('%s/%s-%s-%s-profit-map.pdf' % (figbase, fieldname, year, commodity) , format='pdf')


                        # Profit/Loss Histogram
                        fig = plt.figure()
                        plt.title('%s %s Profit/Loss Distribution' % (year, commodity) )
                        a = np.ravel(ma.compressed(profit / areas))
                        Y,X = np.histogram(a, 10, normed=0, weights=(np.ones(a.shape) * grid_cell_area))

                        def pmap(x):
                            if x < 0:
                                return cmap(X.min())
                            else:
                                return cmap(X.max())

                        x_span = X.max()-X.min()
                        C = [pmap(x) for x in X]
                        plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
                        plt.xlabel('Profit/Loss ($/Acre)')
                        plt.ylabel('Acres')
                        fig.savefig('%s/%s-%s-%s-profit-histogram.pdf' %(figbase, fieldname, year, commodity), format='pdf')

                        # A map distinguishing what is (or is not) profitable.
                        fig = plt.figure()
                        plot_grid_map(profit > 0, cmap=cmap)
                        plt.title('%s %s Profit or Loss' % (year,commodity))
                        fig.savefig('%s/%s-%s-%s-profit-or-loss.pdf' %(figbase, fieldname, year, commodity), format='pdf')


                    except Exception as e:
                        print e
                        print yld.shape
                        print ma.min(yld)
                        print ma.max(yld)
                        print '%s-%s' % (np.min(yld), np.max(yld))

                        print 'Could not generate maps for %s/%s/%s' % (fieldname, year, commodity)
                        # raise e


            # Field summary stats

            # First, Profits.
            vals = ma.dstack(profits)
            mvals = ma.mean(vals, axis=2)
            vvars = ma.std(vals, axis=2)


            fig = plt.figure()
            plt.title('Profit, mean across years ($/Acre)')
            norm = mpl.colors.BoundaryNorm(np.linspace(mvals.min(),mvals.max(),10), cmap.N)
            plot_grid_map(mvals, cmap=cmap, norm=norm)
            plt.colorbar(shrink=0.5)
            fig.savefig('%s/%s-profit-mean.pdf' %(figbase, fieldname))

            fig = plt.figure()
            plt.title('Profit, std deviation across years')
            norm = mpl.colors.BoundaryNorm(np.linspace(vvars.min(),vvars.max(),10), cmap.N)
            plot_grid_map(vvars, norm=norm, cmap=cmap)
            plt.colorbar(shrink=0.5)
            fig.savefig('%s/%s-profit-std.pdf' %(figbase, fieldname))


            fig = plt.figure()
            plt.title('Profit, max across years ($/Acre)')
            ymax = ma.max(vals,axis=2)
            norm = mpl.colors.BoundaryNorm(np.linspace(ymax.min(),ymax.max(),10), cmap.N)
            plot_grid_map(ymax, norm=norm, cmap=cmap)
            plt.colorbar(shrink=0.5)
            fig.savefig('%s/%s-profit-max.pdf' %(figbase, fieldname))


            fig = plt.figure()
            plt.title('Profit, min across years ($/Acre)')
            ymin = ma.min(vals,axis=2)
            norm = mpl.colors.BoundaryNorm(np.linspace(ymin.min(),ymin.max(),10), cmap.N)
            plot_grid_map(ymin, norm=norm, cmap=cmap)
            plt.colorbar(shrink=0.5)
            fig.savefig('%s/%s-profit-min.pdf' % (figbase, fieldname))


            # Now yields
            vals = ma.dstack(ylds)
            mvals = ma.mean(vals, axis=2)
            vvars = ma.std(vals, axis=2)

            fig = plt.figure()
            plt.title('Normalized Yield, mean across years')
            norm = mpl.colors.BoundaryNorm(np.linspace(mvals.min(),mvals.max(),10), cmap.N)
            plot_grid_map(mvals, cmap=cmap, norm=norm)
            plt.colorbar(shrink=0.5)
            fig.savefig('%s/%s-yield-mean.pdf' %(figbase, fieldname))

            fig = plt.figure()
            plt.title('Normalized Yield, std deviation across years')
            norm = mpl.colors.BoundaryNorm(np.linspace(vvars.min(),vvars.max(),10), cmap.N)
            plot_grid_map(vvars, norm=norm, cmap=cmap)
            plt.colorbar(shrink=0.5)
            fig.savefig('%s/%s-yield-std.pdf' %(figbase, fieldname))


            fig = plt.figure()
            plt.title('Normalized Yield, max across years')
            ymax = ma.max(vals,axis=2)
            norm = mpl.colors.BoundaryNorm(np.linspace(ymax.min(),ymax.max(),10), cmap.N)
            plot_grid_map(ymax, norm=norm, cmap=cmap)
            plt.colorbar(shrink=0.5)
            fig.savefig('%s/%s-yield-max.pdf' %(figbase, fieldname))


            fig = plt.figure()
            plt.title('Normalized Yield, min across years')
            ymin = ma.min(vals,axis=2)
            norm = mpl.colors.BoundaryNorm(np.linspace(ymin.min(),ymin.max(),10), cmap.N)
            plot_grid_map(ymin, norm=norm, cmap=cmap)
            plt.colorbar(shrink=0.5)
            fig.savefig('%s/%s-yield-min.pdf' % (figbase, fieldname))



            # Log the number of years of data that we have.
            fig = plt.figure()
            plt.title('Number of years of data')
            cnt = ma.count(vals,axis=2)
            norm = mpl.colors.BoundaryNorm(np.linspace(0,cnt.max(),2*(cnt.max()+1)), cmap.N)
            plot_grid_map(cnt, norm=norm, cmap=cmap)
            plt.colorbar(shrink=0.5)
            fig.savefig('%s/%s-number-years-map.pdf' % (figbase, fieldname))
