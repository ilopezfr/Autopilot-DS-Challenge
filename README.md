## Autopilot Coding Challenge
-----
This repository contains the code used to solve the problem described below. 

## Contents
- `tesla_program.py` main program to run through the different questions. 
- `utils.py` a collection of helper functions saved as module for reuse.
- `data` folder containing the files with 'csv' extension each corresponding to an individual car data.

## Requirements:
Python 3 is required along with the following modules:
- numpy
- pandas
- scikit-learn (to use k-means clustering algorithm)
```
cd src
pip install -r requirements
```
## Solution:
**Question 1.1**: Compute the 5 least common values by total occurence
```
python tesla_program.py -q 1.1 -n 5 -a true
```
Big-O complexity: N + K*log(k); where N=row number, K=distinct values. 
Run time: 278.969 ms

**Question 1.2**: Compute the 3 most common values by total time spent at that value
```
python tesla_program.py -q 1.2 -n 3 -a false
```

Big-O complexity: N + K*log(k); where N=row number, K=distinct values. 
Run time: 406.112 ms

**Question 1.3**: Compute the 3 largest cycles. Return the amplitud of the cycle, value of the local minima, value of the local maxima, and the length of time between the minima and the maxima.
```
python tesla_program.py -q 1.3 -n 3 -a false
```

Big-O complexity: N + K*log(k); where N=row number, K=distinct values. 
Run time: 300.283 ms

**Question 2**: Develop a generalized method to programmatically differentiate vehicles that are behaving differently due to higher damage accrual. 
```
python tesla_program.py -q 2
```

Run time: 505983.279 ms


## Notes
For **Question 1.2** for each measurement (m) at a Timestep (t), I considered  the time spent is the time difference between the current timestep (t) and the previous timestep (t-1).

For **Question 1.3** a cycle is defined as a progression from a given local minima/maxima to the next immediate local minima/maxima. The size of a cycle is determined by the amplitude of that cycle.

The chart below represents a 1-minute extract from car 0 signal measurements. A cycle is the distance in y-axis from a local minima (red dot) to a local maxima (green dot).

![alt text][image3]

For **Question 2** I considered different methodologies such as Dynamic Time Warping,  and Time Series Motifs. 

Due to time constraints I chose a methodology I was most familiar with. To programmatically identify the cars that are not behaving properly, I used Anomaly Detection using K-means clustering of Time Series. While in practice this methodology may not be the most robust compared with other options, it gives ....

The algorithm works as follows:.

... 

---

## Problem Statement
The Autopilot Data Infrastructure team frequently works with timeseries signal data coming from vehicles across our fleet. We  analyze these signals by extracting specific metrics to help us understand how users are interacting with the vehicle. Example metrics include: the most or least common values seen, the most or least time spent at given values, or the largest or smallest deltas seen in the signal.

## Question I
##### For this question, only use data for car_0 - this question should take ~40% of your time
To that end, build a module that provides functionality to operate on a timeseries and allows for the following queries: 
 - The top N most/least common values by total occurrence
 - The top N most/least common values by total time spent
 - The top N largest/smallest 'cycles' - we will define a cycle as a progression from a given local minima/maxima to the next immediate local minima/maxima. The size of a cycle is determined by the amplitude of that cycle.

 Using this module and the sample data provided, please compute the following:
 - The 5 least common values by total occurence
 - The 3 most common values by total time spent at that value
 - The 3 largest cycles (largest delta between the minima and maxima points). When returning a cycle, please provide the value of the minima, the value of the maxima, and the amplitude of the cycle. Bonus if you can also provide the length of time between the minima and maxima


## Question II
##### For this question, use data for car_0 - car_9 - this question should take ~60% of your time

We have provided drive unit torque data for 10 vehicles in the form of 10 timeseries. Two of these vehicles have experienced higher damage accrual than the other 8. Determine which 2 vehicles are the ones behaving differently and develop a generalized method to programmatically differentiate these 2 vehicles from the other 8.      

## Deliverable:
Provide directions to run your code and to recreate the solutions to the above questions. This would include installing dependencies, specifying a path, or running the executable. Python is the standard expected language for this coding challenge. You may use any library. However, if using a library outside those available in the standard python installation (numpy, pandas, collections), make a note of it and explain why it is necessary.

Provide solutions along with runtimes for each question.
For any modules or funtions that you provide, document the Big-O complexity of each API call in your class as a function of the length of a given timeseries N.

------

[//]: # (Image References)

[image0]: ./images/car_0_good.png "Car 0"
[image1]: ./images/car_3_bad.png "Car 3"
[image2]: ./images/library_of_shapes.png "Segments of Time Series"
[image3]: ./images/min_max_cycles.png "Cycles"