import pandas as pd
import numpy as np
from glob import glob
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)


## For Question 1

# 1. The top N most/least common values by total occurrence

def top_N_common(df, n=5, ascending=False):
    print(ascending)
    count_by_value = (df.groupby("sig_value").agg({"timestamp_utc":"count"}).sort_values(by="timestamp_utc", ascending=ascending).reset_index())
    return count_by_value.head(n).sig_value  #.tolist()



# 2. Top N most/least common values by total time spent
# assuming that for each measurement (m) at a Timestep (t), the time spent 
# is the time difference between the current timestep (t) and the previous timestep (t-1).

def top_N_common_by_time(df, n=3, ascending=False):

    df['time_spent'] = (pd.to_datetime(df['timestamp_utc']) 
                                  -pd.to_datetime(df['timestamp_utc'].shift(1)))/ np.timedelta64(1, 's')

    count_by_time_spent = (df.groupby("sig_value").agg({'time_spent':'sum'}).sort_values(by="time_spent", ascending=ascending).reset_index())
    return count_by_time_spent.head(10).sig_value



# 3. Top N largest/smallest cycles

def largest_cycle(df, n=5, ascending = False):
    # Local min indexes: previous value and next value should be higher than current value. 
    min_values = ((df.sig_value.shift(1) > df.sig_value) 
                                & (df.sig_value.shift(-1) > df.sig_value)) 

    # Local max indexes: previous value and next value should be smaller than current value. 
    max_values = ((df.sig_value.shift(1) < df.sig_value) 
                                & (df.sig_value.shift(-1) < df.sig_value)) 

    df['min'] = df.loc[min_values, 'sig_value']
    df['max'] = df.loc[max_values, 'sig_value']
    #df['min_max'] = df.loc[max_values|min_values, 'sig_value']
    df_dense = df[df['min'].notnull() | df['max'].notnull()].reset_index()
    df_dense['amplitude_cycle'] = abs(df_dense['sig_value'].shift(1) - df_dense['sig_value'] )
    df_dense['time_spent_cycle_sec'] = (pd.to_datetime(df_dense['timestamp_utc']) 
                                  -pd.to_datetime(df_dense['timestamp_utc'].shift(1)))/ np.timedelta64(1, 's')
    df_dense['min'] = df_dense['min'].fillna(df_dense['sig_value'].shift(1))
    df_dense['max'] = df_dense['max'].fillna(df_dense['sig_value'].shift(1))

    return df_dense.loc[:, ['amplitude_cycle', 'time_spent_cycle_sec' ,'min','max' ]].dropna().sort_values(by='amplitude_cycle', ascending=ascending).head(n)



## For Question 2

# Read data files
# def read_data(files):
#     """read cars data from the directory of csv files"""
#     path = path
#     files = glob(path + '/*.csv')
#     get_df = lambda f: pd.read_csv(f)
#     all_cars = {f: get_df(f) for f in files}
#     return all_carsdef read_data(files):

def sliding_segments(data, segment_len, slide_len):
    """ Split time series data into segments  
        segment_len: how many consecutive value in each segment
        slide_len: sliding window size
    """
    segments = []
    for start_pos in range(0, len(data), slide_len):
        end_pos = start_pos + segment_len
        # make a copy so changes to 'segments' doesn't modify the original data
        segment = np.copy(data[start_pos:end_pos])
        # if the last segment is truncated, drop it. 
        if len(segment) != segment_len:
            continue
        segments.append(segment)
    return segments


def get_windowed_segments(segments, segment_len, window):
    """
    Apply a window function to the list of all segments, 
    which forces the start and end of each segment to be zero
    """
    windowed_segments = []
    for segment in segments:
        segment *= window
        windowed_segments.append(segment)
    return windowed_segments


# Reconstruction from clusters:
# Reconstructing our signal to be tested using the learned library of shapes. 
# - Split the data into overlapping segments
# - Find the cluster centroid which best matches our segment
# - Use that centroid as the reconstruction for that segment
# - Join the reconstruction segments up to form the reconstruction

def reconstruct(data, window, clusterer):
    """
    Reconstruct the data using the centroids from clusterer (kmeans)
    """
    
    segment_len = len(window)
    slide_len = round(segment_len/2)
    segments = sliding_segments(data, segment_len, slide_len)
    reconstructed_data = np.zeros(len(data))
    for segment_n, segment in enumerate(segments):
        # window the segment so that we can find it in our clusters which were
        # formed from windowed data
        segment *= window
        nearest_centroid_idx = clusterer.predict(segment)[0]
        nearest_centroid = np.copy(clusterer.cluster_centers_[nearest_centroid_idx])

        pos = segment_n * slide_len
        reconstructed_data[pos:pos+segment_len] += nearest_centroid

    return reconstructed_data






