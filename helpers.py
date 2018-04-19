import warnings
import numpy as np
from functools import reduce
from sklearn.preprocessing import scale


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


def make_tensors(df):
    # for each band, stack the normalized img arrays on top of each other
    norm_band_1 = np.stack(scale(arr) for arr in np.array(df.band_1))
    norm_band_2 = np.stack(scale(arr) for arr in np.array(df.band_2))
    norm_band_3 = np.stack(scale(arr) for arr in np.array(df.band_3))
    # combine the normalized bands into three channels
    flat_tensors = np.stack([norm_band_1, norm_band_2, norm_band_3], axis=-1)
    # return tensors reshaped into 75x75 radar images
    return flat_tensors.reshape(flat_tensors.shape[0], 75, 75, 3)


def L(x):
    return 1 + np.log(x) if x > 1 else x


def bentes_norm(arr):
    """"""
    Larr = np.fromiter((L(x) for x in arr), dtype=arr.dtype,
                       count=reduce(lambda a, b: a * b, arr.shape))
    maxL = abs(Larr.max())
    return Larr / maxL
