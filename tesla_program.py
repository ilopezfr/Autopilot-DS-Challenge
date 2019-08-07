import sys, os, argparse, time, glob, warnings
import pandas as pd
import numpy as np
import utils
from sklearn.cluster import KMeans


def q11(path, n=5, ascending=False):
    start = time.time()
    filepath = os.path.abspath(os.path.join(path, 'car_0.csv'))
    #print('looking for {}'.format(filepath))
    df = pd.read_csv(filepath) 
    print(utils.top_N_common(df,n, ascending))
    end = time.time()
    print('Function took {:.3f} ms'.format((end-start)*1000.0))

def q12(path, n=3, ascending=False):
    start = time.time()
    filepath = os.path.abspath(os.path.join(path, 'car_0.csv'))
    df = pd.read_csv(filepath) 
    print(utils.top_N_common_by_time(df,n, ascending))
    end = time.time()
    print('Function took {:.3f} ms'.format((end-start)*1000.0))

def q13(path, n=3, ascending=False):
    start = time.time()
    filepath = os.path.abspath(os.path.join(path, 'car_0.csv'))
    df = pd.read_csv(filepath)
    print(utils.largest_cycle(df,n, ascending))
    end = time.time()
    print('Function took {:.3f} ms'.format((end-start)*1000.0))

def q2(path):
    print("Reading data...")
    start = time.time()
    #path = path
    filepath = os.path.abspath(path)
    #print(filepath)
    get_df = lambda f: pd.read_csv(f)
    all_cars = {f: get_df(os.path.join(filepath, f)) for f in os.listdir(filepath)}
    print(len(all_cars))

    print('Preparing data for clustering...')
    # select signal time series data from cars behaving correctly
    # (ignoring those that don't) and append it.
    signal_values = []
    for k,v in all_cars.items():
        car_num = int(k.split('/')[-1].split('.')[0][-1])
        anomalous_cars = [3,7]  # list the cars that we won't be included for train
        if car_num not in anomalous_cars:
            #print(car_num)
            car_sig_value = np.array(v.sig_value)  #[100:100100]) # pick 100k signal values for each 
            signal_values.extend(car_sig_value)
            #len(signal_values)
        else:
            pass
    print('Length signal_values:', len(signal_values))
    
    # split signal data into segments. 
    # setup the segment length and slide length parameters
    segment_len = 100  # try with 100
    slide_len = 10  # try with 10
    segments = utils.sliding_segments(signal_values, segment_len, slide_len)
    print("Produced %d signal values segments" % len(segments))   # 10763
    
    # apply a window function to the data, which forces the start and end to be zero
    window_rads = np.linspace(0, np.pi, segment_len)
    window = np.sin(window_rads)**2
    windowed_segments = utils.get_windowed_segments(segments,segment_len , window)
    #print(len(windowed_segments))

    # Apply k-means clustering on the segments
    print('Clustering...')
    k = 150 # Test different n_clusters
    clusterer = KMeans(n_clusters=k)  
    clusterer.fit(windowed_segments)

    # Reconstruct the data using the centroids from the clusterer calculated
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    print('Reconstructing...')    
    reconstructed = utils.reconstruct(signal_values, window, clusterer)
 
    # Anomaly detection:
    print('Anomaly Detection...') 
    for k,v in all_cars.items():
        car_num = int(k.split('/')[-1].split('.')[0][-1])
        car_signal_data = v.sig_value
        #print(car_signal_data.shape)
        reconstructed = utils.reconstruct(car_signal_data, window, clusterer)
        error = reconstructed - car_signal_data
        error_98th_percentile = np.percentile(error, 98)
        #print("Car %d Maximum reconstruction error was %.1f" % (car_num, error.max()))
        #print("Car %f 98th percentile of reconstruction error was %.1f" % (car_num,error_98th_percentile))
        # Car 1 98th percentile of reconstruction error was 106.5
        # Car 8 98th percentile of reconstruction error was 103.0
        # Car 5 98th percentile of reconstruction error was 73.0
        # Car 7 98th percentile of reconstruction error was 318.9
        # Car 2 98th percentile of reconstruction error was 110.6
        # Car 9 98th percentile of reconstruction error was 113.7
        # Car 0 98th percentile of reconstruction error was 103.7
        # Car 6 98th percentile of reconstruction error was 110.0
        # Car 3 98th percentile of reconstruction error was 340.8
        # Car 4 98th percentile of reconstruction error was 103.2
        error_threshold = 150 # Avg. 98th percentile Reconstruction Error is ~105.
        if error_98th_percentile > error_threshold:
            print("Car %d not performing appropriately due to potential damage. Needs revison."%(car_num))
        else:
            pass


    end = time.time()
    print('Function took {:.3f} ms'.format((end-start)*1000.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Accept arguments for questions')
    parser.add_argument('-q', action='store',  required=True, dest='question')
    parser.add_argument('-d', '--data', action='store', default='data', required=False, dest='data')
    parser.add_argument('-n', action='store', default = 5, required=False, dest='top_n')
    parser.add_argument('-a', '--ascending', action='store', default = 'false', choices=['true','false'], required=False, dest='ascending')

    args = parser.parse_args()

    topN = int(args.top_n)
    asc = args.ascending == 'true'

    if args.question == '1.1':
        q11(args.data, topN, asc)
    if args.question == '1.2':
        q12(args.data, topN, asc)
    if args.question == '1.3':
        q13(args.data, topN, asc)
    if args.question == '2':
        q2(args.data)
    

