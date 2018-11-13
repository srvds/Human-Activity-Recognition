2018 november

Human-Activity-Recognition Using Smartphones Data Set 
===========================


---------------------------------------------------------------------------

Repository Overview:
--------------------

This project aims to build a model that predicts the human activities such as Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing and Laying from the Sensor data of smart phones.
<br><br>
The repository has 3 ipython notebook
1 [HAR_EDA.ipynb](https://github.com/srvds/Human-Activity-Recognition/blob/master/HAR_EDA.ipynb) Data pre-processing and Exploratory Data Analysis
<br>
2 [HAR_PREDICTION_MODELS.ipynb](https://github.com/srvds/Human-Activity-Recognition/blob/master/HAR_PREDICTION_MODELS.ipynb) Machine Learning models with featured data
<br>
3 [HAR_LSTM.ipynb](https://github.com/srvds/Human-Activity-Recognition/blob/master/HAR_LSTM.ipynb) LSTM model on raw timeseries data
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

Every modern Smart Phone has a number of [sensors](https://www.gsmarena.com/glossary.php3?term=sensors).we are interested in two of the sensors Accelerometer and Gyroscope.
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
### Understanding the dataset
<br>
Both sensors generate data in 3 Dimensional space over time. Hence data the captured '3-axial linear acceleration'(_tAcc-XYZ_) from accelerometer and '3-axial angular velocity' (_tGyro-XYZ_) from Gyroscope with several variations<br>
prefix 't' in those metrics denotes time.<br>
suffix 'XYZ' represents 3-axial signals in X , Y, and Z directions.<br>
The available data is pre-processed and featurized by signal processing experts

