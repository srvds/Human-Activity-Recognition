2018 november

Human-Activity-Recognition Using Smartphones Data Set 
===========================


---------------------------------------------------------------------------

Repository Overview:
--------------------

This project aims to build a model that predicts the human activities such as Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing and Laying from the Sensor data of smart phones.
<br><br>
The repository has 3 ipython notebook
<br>
1 [HAR_EDA.ipynb](https://github.com/srvds/Human-Activity-Recognition/blob/master/HAR_EDA.ipynb) : Data pre-processing and Exploratory Data Analysis
<br>
2 [HAR_PREDICTION_MODELS.ipynb](https://github.com/srvds/Human-Activity-Recognition/blob/master/HAR_PREDICTION_MODELS.ipynb) : Machine Learning models with featured data
<br>
3 [HAR_LSTM.ipynb](https://github.com/srvds/Human-Activity-Recognition/blob/master/HAR_LSTM.ipynb) : LSTM model on raw timeseries data
<br><br>
All the code is in python 3 <br>
**DEPENDENCIES**
* tensorflow
* keras
* numpy
* pandas
* matplotlib
* seaborn
* sklearn
* itertolls
* datetime


Introduction:
-------------

Every modern Smart Phone has a number of [sensors](https://www.gsmarena.com/glossary.php3?term=sensors). we are interested in two of the sensors Accelerometer and Gyroscope.
<br>
The data is recorded with the help of sensors
<br>
This is a 6 class classification problem as we have 6 activities to detect.<br>

This project has two parts, the first part trains, tunes and compares Logistic Regression, Linear support vector classifier, RBF(Radial Basis Function) SVM classifier, Decision Tree, Random Forest, Gradient Boosted Decision Trees  model and uses the data featured by domain expert.<br>
The second part uses the raw time series windowed data to train (Long Short term Memory)LSTM models. The LSTM models are semi tuned manually to fast forward the tuning task.

-----------------------------------------------------

Dataset:
--------

The dataset can be downloaded from
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones#
<br><br>
dataset is also included in the Repository with in the folder [UCI_HAR_Dataset](https://github.com/srvds/Human-Activity-Recognition/tree/master/UCI_HAR_Dataset)
<br><br>
Human Activity Recognition database is built from the recordings of 30 persons performing activities of daily living (ADL) while carrying a waist-mounted smartphone with embedded inertial sensors(accelerometer and Gyroscope).
**Activities**
* Walking
* Walking Upstairs
* Walking Downstairs
* Sitting
* Standing
* Laying

[**Accelerometers**](https://en.wikipedia.org/wiki/Accelerometer) detect magnitude and direction of the proper acceleration, as a vector quantity, and can be used to sense orientation (because direction of weight changes)
<br><br>
[**GyroScope**](https://en.wikipedia.org/wiki/Gyroscope) maintains orientation along a axis so that the orientation is unaffected by tilting or rotation of the mounting, according to the conservation of angular momentum.
<br><br>
Accelerometer measures the directional movement of a device but will not be able to resolve its lateral orientation or tilt during that movement accurately unless a gyro is there to fill in that info.
<br>
With an accelerometer you can either get a really "noisy" info output that is responsive, or you can get a "clean" output that's sluggish. But when you combine the 3-axis accelerometer with a 3-axis gyro, you get an output that is both clean and responsive in the same time.
<br><br>
#### Understanding the dataset
* Both sensors generate data in 3 Dimensional space over time. Hence the data captured are '3-axial linear acceleration'(_tAcc-XYZ_) from accelerometer and '3-axial angular velocity' (_tGyro-XYZ_) from Gyroscope with several variations.
* prefix 't' in those metrics denotes time.
* suffix 'XYZ' represents 3-axial signals in X , Y, and Z directions.
* The available data is pre-processed by applying noise filters and then sampled in fixed-width windows(sliding windows) of 2.56 seconds each with 50% overlap. ie., each window has 128 readings.
#### Featurization
For each window a feature vector was obtained by calculating variables from the time and frequency domain. each datapoint represents a window with different readings.<br>
Readings are divided into a window of 2.56 seconds with 50% overlapping. 
* Accelerometer readings are divided into gravity acceleration and body acceleration readings,
  which has x,y and z components each.

* Gyroscope readings are the measure of angular velocities which has x,y and z components.

* Jerk signals are calculated for BodyAcceleration readings.

* Fourier Transforms are made on the above time readings to obtain frequency readings.

* Now, on all the base signal readings., mean, max, mad, sma, arcoefficient, engerybands,entropy etc., are calculated for each window.

* We get a feature vector of 561 features and these features are given in the dataset.

* Each window of readings is a datapoint of 561 features,and we have 7352, 

* These are the signals that we got so far.(prefix t means time domain data, prefix f means frequency domain data)
#### Train and test data were saperated
 - The readings from ___70%___ of the volunteers(21 people) were taken as ___trianing data___ and remaining ___30%___ volunteers recordings(9 people) were taken for ___test data___
* All the data is present in 'UCI_HAR_dataset/' folder in present working directory.
     - Feature names are present in 'UCI_HAR_dataset/features.txt'
     - ___Train Data___
         - 'UCI_HAR_dataset/train/X_train.txt'
         - 'UCI_HAR_dataset/train/subject_train.txt'
         - 'UCI_HAR_dataset/train/y_train.txt'
     - ___Test Data___
         - 'UCI_HAR_dataset/test/X_test.txt'
         - 'UCI_HAR_dataset/test/subject_test.txt'
         - 'UCI_HAR_dataset/test/y_test.txt'
 
 -------------------------------------------------------------------------------
 
 Analysis
 --------
 For detailed code of this section you can always check the [HAR_EDA Notebook](https://github.com/srvds/Human-Activity-Recognition/blob/master/HAR_EDA.ipynb)
 <br><br>
 #### Check for Imbalanced class
 <br>
 if some class have too little or too large numbers of values compared to rest of the classes than the dataset is imbalanced.
 **Plot-1**

 
<img src="https://github.com/srvds/Human-Activity-Recognition/blob/master/plots/plot1.png" height=500 width=700>
<br><br>
In this plot on the X-axis we have subjects(volunteers) 1 to 30. Each color represents an activity<br>
on the y-axis we have amount of data for each activity by provided by each subject.<br>
<br>
**Plot-2**
<br>

<img src="https://github.com/srvds/Human-Activity-Recognition/blob/master/plots/plot2.png">
<br><br>
From plot1 and plot2 it is clear that dataset is almost balanced.<br>

#### check for ease of classification
<br>
**Plot-3**
<img src="https://github.com/srvds/Human-Activity-Recognition/blob/master/plots/plot3.png">
