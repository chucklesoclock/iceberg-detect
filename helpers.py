import warnings
import numpy as np


def process_df(df):
    # reclassify bands as numpy array
    if not isinstance(df.band_1.iloc[0], np.ndarray):
        for band in ['band_1', 'band_2']:
            df[band] = df[band].apply(np.asarray)
            
    # set band_3 as mean of first two bands
    if 'band_3' not in df.columns:
        df['band_3'] = (df.band_1 + df.band_2) / 2
        print('Third band added!')
    else:
        print('Third band already present')
    
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        try:
            # replace 'na' with np.nan in inc_angle
            df.loc[df['inc_angle'] == 'na', 'inc_angle'] = np.nan
            # reclassify incidence angle as floats
            df['inc_angle'] = df['inc_angle'].astype('float64')
            print('inc_angle reclassified as floats with NaNs!')
        except TypeError:
            print('inc_angle already all floats')