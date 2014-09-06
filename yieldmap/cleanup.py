def cleanup(df):

    # Only keep 'mappable' points
    mappable = df[df.Mappable == 1]

    # Drop duplicate positions
    nodups = mappable.drop_duplicates(cols=['Latitude', 'Longitude'])

    # Calculate area and speed, only keep those points with positive area.
    nodups['Area'] = nodups.Distance * nodups.Width / 6272640 # Area in acres
    nodups['Speed'] = nodups.Distance / nodups.Duration
    nodups = nodups[nodups.Area > 0]

    nodups['Yield'] = nodups.apply(calculate_yield, axis=1)

    # remove_outliers removes samples > 3 std. deviations from the
    # mean, for the various variables.
    def remove_outliers(df, key):
        grouped = df.groupby(['Season', 'Field', 'Commodity'])
        return df[abs(grouped[key].transform(zscore)) < 3]

    distance_valid = remove_outliers(nodups, 'Distance')
    moisture_valid = remove_outliers(distance_valid, 'Moisture_s')
    mass_valid = remove_outliers(moisture_valid, 'Mass_Flow_')
    yield_valid = remove_outliers(mass_valid, 'Yield')



def zscore(x):
    return (x - x.mean()) / x.std()

# Conversion of wet weight to market weight. Table has market moisture, market weight (lbs/bu)
market = { 'SOYBEANS': (13.0, 60.0),
           'CORN': (15.0, 55.0),
           'CORN 2': (15.0, 55.0),
           'CORN 4': (15.0, 55.0)
         }


def calculate_yield(s):
    (mmass, mbushel) = market[s['Commodity']]
    adjustment = (100.0 - s['Moisture_s']) / (100.0 - mmass)
    yld =  ((adjustment * s['Mass_Flow_'] * s['Duration']) / mbushel) / s['Area']
    return yld
