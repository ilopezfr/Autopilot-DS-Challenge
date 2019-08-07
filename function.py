

## Attempt to do it using lists:
import csv
import cv2
lines=[]
with open('data/car_0.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)



df_sample['max'] = df_sample.loc[(df_sample.sig_value.shift(1) < df_sample.sig_value) 
                            & (df_sample.sig_value.shift(-1) < df_sample.sig_value), 'sig_value']



for row in lines 

##################################

import pandas as pd
import numpy as np

df = pd.read_csv('data/car_0.csv')

# 1. The top N most/least common values by total occurrence
def top_N_common(df, n=5, ascending=False):
    count_by_value = (df.groupby("sig_value").agg({"timestamp_utc":"count"}).sort_values(by="timestamp_utc", ascending=ascending).reset_index()
                      )
    return count_by_value.head(n).sig_value  #.tolist()
                      
# test: The 5 least common values by total occurence
%time 
top_N_common(df,5, ascending=True)

# 2. Top N most/least common values by total time spent
## Assuming discrete sig_values, where time spent in timestep t is calculated by the different of the interval t-1 and t. 

def top_N_common_by_time(df, n=3, ascending=False):

    df['time_spent'] = (pd.to_datetime(df['timestamp_utc']) 
                                  -pd.to_datetime(df['timestamp_utc'].shift(1)))/ np.timedelta64(1, 's')

    count_by_time_spent = (df.groupby("sig_value").agg({'time_spent':'sum'}).sort_values(by="time_spent", ascending=ascending).reset_index())
    return count_by_time_spent.head(10).sig_value #, count_by_time_spent.head(10).time_spent

# test: The 3 most common values by total time spent at that value
%time
top_N_common_by_time(df, n=5, ascending=False)   


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
    df['min_max'] = df.loc[max_values|min_values, 'sig_value']
    df_dense = df[df['min'].notnull() | df['max'].notnull()].reset_index()
    df_dense['amplitude_cycle'] = abs(df_dense['sig_value'].shift(1) - df_dense['sig_value'] )
    df_dense['time_spent_cycle_sec'] = (pd.to_datetime(df_dense['timestamp_utc']) 
                                  -pd.to_datetime(df_dense['timestamp_utc'].shift(1)))/ np.timedelta64(1, 's')
    df_dense['min'] = df_dense['min'].fillna(df_dense['sig_value'].shift(1))
    df_dense['max'] = df_dense['max'].fillna(df_dense['sig_value'].shift(1))

    return df_dense.loc[:, ['amplitude_cycle', 'time_spent_cycle_sec' ,'min','max' ]].dropna().sort_values(by='amplitude_cycle', ascending=ascending).head(n)
    

# test: The 3 largest cycles 
%time 
largest_cycle(df, 3, ascending=False)   


####
# Exploratory Data Analysis

df_1min_sample = df[(pd.to_datetime(df['date']) == '2017-09-01') & 
                   (df['hour'] == 2) &
                    (df['minute'] == 5) #& (df['minute'] < 7)
                      ]
df_1min_sample.head()
df_1min_sample.plot(x='time', y='sig_value' )

##  Test to identify the local min and local max of the time series. 
# Local min indexes: previous value and next value should be higher than current value. 
min_values = ((df_sample.sig_value.shift(1) > df_sample.sig_value) 
                            & (df_sample.sig_value.shift(-1) > df_sample.sig_value)) 

# Local max indexes: previous value and next value should be smaller than current value. 
max_values = ((df_sample.sig_value.shift(1) < df_sample.sig_value) 
                            & (df_sample.sig_value.shift(-1) < df_sample.sig_value)) 
df_sample = df_sample.copy()
df_sample['min'] = df_sample.loc[min_values, 'sig_value']
df_sample['max'] = df_sample.loc[max_values, 'sig_value']
df_sample['min_max'] = df_sample.loc[max_values|min_values, 'sig_value']

df_sample.head()

# Plot results
plt.scatter(df_sample.index, df_sample['min'], c='r')
plt.scatter(df_sample.index, df_sample['max'], c='g')
df_sample.sig_value.plot()


##############

#1 : Pull all 8 cars that behave properly. 

path = './data/'
files = glob(path + '/*.csv')

def read_data(files):
    """read cars data from a directory of files"""
    get_df = lambda f: pd.read_csv(f)
    all_cars = {f: get_df(f) for f in files}
    return all_cars

all_cars = read_data(files)

#car_0 = all_cars['./data/car_0.csv']
#s0 = np.array(car_0.sig_value)

# select signal time series data from cars behaving correctly and append it 
signal_values = []
for k,v in all_cars.items():
    car_num = int(k.split('/')[-1].split('.')[0][-1])
    anomalous_cars = [3,7]
    if car_num not in anomalous_cars:
        #print(car_num)
        car_sig_value = np.array(v.sig_value)  #[100:100100]) # pick 100k signal values for each 
        signal_values.extend(car_sig_value)
        #len(signal_values)
    else:
        pass
# Now we should have 1077284 signal points that we can use to construct our k-means


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

segment_len = 1000  # try with 100
slide_len = 100  # try with 10

segments = sliding_segments(signal_values, segment_len, slide_len)
print("Produced %d signal values segments" % len(segments))   # 10763

# apply a window function to the data, which forces the start and end to be zero
window_rads = np.linspace(0, np.pi, segment_len)
window = np.sin(window_rads)**2
#plt.plot(window)
#plt.show()

def get_windowed_segments(segments, segment_len):
    """
    Apply a window function to the list of all segments, 
    which forces the start and end of each segment to be zero
    """
    window_rads = np.linspace(0, np.pi, segment_len)
    window = np.sin(window_rads)**2
    windowed_segments = []
    for segment in segments:
        segment *= window
        windowed_segments.append(segment)
    return windowed_segments

windowed_segments = get_windowed_segments(segments)


## Clustering
from sklearn.cluster import KMeans

clusterer = KMeans(n_clusters=50)  # Test different n_clusters
clusterer.fit(windowed_segments)

## Reconstruction from clusters:
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

reconstructed = reconstruct(signal_values, window, clusterer)

error = reconstructed - signal_values
error_98th_percentile = np.percentile(error, 98)
print("Maximum reconstruction error was %.1f" % error.max())
print("98th percentile of reconstruction error was %.1f" % error_98th_percentile)


#Anomaly detection
# Test for signal data coming from car 3 and car 7
anomalous_signal_values = all_cars['./data/car_3.csv']

reconstructed_anomalous = reconstruct(anomalous_signal_values, window, clusterer)
error = reconstructed_anomalous - data_anomalous
error_98th_percentile = np.percentile(error, 98)
print("Maximum reconstruction error was %.1f" % error.max())
print("98th percentile of reconstruction error was %.1f" % error_98th_percentile)


# TODO: Setup threshold: np.percentile(error, 98) > 250 --> anomalous




def ts_anomaly_detection(df ):
    """
    Perform anomaly detection on time-series value_sig


    """



    if  ... 

    else:
        raise(Exception("Vehicle not performing appropriately. Could have experience damage. Needs revison."))

