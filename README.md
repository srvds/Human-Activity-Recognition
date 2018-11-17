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
All the code is written in python 3 <br><br>
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
<br>
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

* Each window of readings is a datapoint of 561 features,and we have 10299 readings.

* These are the signals that we got so far.(prefix t means time domain data, prefix f means frequency domain data)
#### Train and test data were saperated
 - The readings from ___70%___ of the volunteers(21 people) were taken as ___trianing data___ and remaining ___30%___ volunteers recordings(9 people) were taken for ___test data___
* All the data is present in 'UCI_HAR_dataset/' folder in present working directory.
     - Feature names are present in 'UCI_HAR_dataset/features.txt'
     - ___Train Data___ (7352 readings)
         - 'UCI_HAR_dataset/train/X_train.txt'
         - 'UCI_HAR_dataset/train/subject_train.txt'
         - 'UCI_HAR_dataset/train/y_train.txt'
     - ___Test Data___ (2947 readinds)
         - 'UCI_HAR_dataset/test/X_test.txt'
         - 'UCI_HAR_dataset/test/subject_test.txt'
         - 'UCI_HAR_dataset/test/y_test.txt'
 
-------------------------------------------------------------------------------

Analysis
--------

For detailed code of this section you can always check the [HAR_EDA Notebook](https://github.com/srvds/Human-Activity-Recognition/blob/master/HAR_EDA.ipynb)
<br>
#### Check for Imbalanced class<br>
if some class have too little or too large numbers of values compared to rest of the classes than the dataset is imbalanced.<br>
**Plot-1**
<br>
<img src="https://github.com/srvds/Human-Activity-Recognition/blob/master/plots/plot1.png" height=500 width=700>
<br><br>
In this plot on the X-axis we have subjects(volunteers) 1 to 30. Each color represents an activity<br>
On the y-axis we have amount of data for each activity by provided by each subject.<br>
**Plot-2**
<br>
<img src="https://github.com/srvds/Human-Activity-Recognition/blob/master/plots/plot2.png">
<br><br>
From plot1 and plot2 it is clear that dataset is almost balanced.<br>
#### Variable analysis
**Plot-3**
<br>
``` python
sns.set_palette("Set1", desat=0.80)
facetgrid = sns.FacetGrid(train, hue='ActivityName', size=6,aspect=2)
facetgrid.map(sns.distplot,'tBodyAccMagmean', hist=False)\
    .add_legend()
plt.annotate("Stationary Activities", xy=(-0.956,17), xytext=(-0.9, 23), size=20,\
            va='center', ha='left',\
            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))

plt.annotate("Moving Activities", xy=(0,3), xytext=(0.2, 9), size=20,\
            va='center', ha='left',\
            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))
plt.show()
```
<br>
<img src="https://github.com/srvds/Human-Activity-Recognition/blob/master/plots/plot3.png">
The above plot is of tBodyAccMagmean which is mean values of magnitude of acceleration in time space. <br>

**Plot-4**
<br>Box plot, mean of magnitude of an acceleration
<br>
``` python
plt.figure(figsize=(7,7))
sns.boxplot(x='ActivityName', y='tBodyAccMagmean',data=train, showfliers=False, saturation=1)
plt.ylabel('Acceleration Magnitude mean')
plt.axhline(y=-0.7, xmin=0.1, xmax=0.9,dashes=(5,5), c='g')
plt.axhline(y=-0.05, xmin=0.4, dashes=(5,5), c='m')
plt.xticks(rotation=90)
plt.show()
```
<img src="https://github.com/srvds/Human-Activity-Recognition/blob/master/plots/plot4.png">
<br>
From plot-3 and plot-4 we can see that stationary activities can be linearly separated from activities with motion.
<br>

**Plot-5**
<br>
Dimensionality reduction using T-distributed Stochastic Neighbor Embedding (t-SNE) to visualize 561 dimension dataset.
<br>
<img src="https://github.com/srvds/Human-Activity-Recognition/blob/master/t-sne_perp_50_iter_1000.png" width="600">
<br>
Sitting and standing are overlapped while other 4 classes can be separated well.

------------------------------------------------------------------------------------

Models
------

#### Machine Learning Algorithms

scikit-learn is used for all the 6 alogorithms listed below.<br>
Hyperparameters of all models are tuned by grid search CV<br>
Models fitted:<br>
- Logistic Regression
- Linear Support Vector Classifier(SVC)
- Radial Basis Function (RBF) kernel SVM classifier 
- Decision Tree 
- Random Forest 
- Gradient Boosted DT

#### Models Comparisions
|  model  | Accuracy |  Error|
|---|---|---|
| Logistic Regression |  96.27% | 3.733% |
| Linear SVC | 96.61% |  3.393% |
|rbf SVM classifier  | 96.27%    |  3.733% |
|Decision Tree  |       86.43%   |   13.57% |
|Random Forest |      91.31%    |  8.687% |
|Gradient Boosted DT | 91.31%    |  8.687% |


> **Observing the Top 2 Models**

**Logistic Regression**

**Plot-6**

Normalized confusion matrix for Linear Regression Model

<img src="https://github.com/srvds/Human-Activity-Recognition/blob/master/plots/plot7.png">

Diagonal Value of 1 means 100% accuracy for that class, and 0 means 0% accuracy.<br>
considering the diagonal elements we have value 1 for rows corresponding to 'Laying' and 'Walking'.<br>
while 'sitting' has value of only 0.87. In the row 2nd row and 3rd column we have value 0.12 which basically means about 12% readings of the class sitting is misclassified as standing.

**Linear SVC**

**Plot-7**

Normalized confusion matrix for Linear SVC Model

<img src="https://github.com/srvds/Human-Activity-Recognition/blob/master/plots/plot8.png">

In this model also the diagonal elements, we have value 1 for rows corresponding to 'Laying' and 'Walking'.<br>
Again row corresponding to 'sitting' has value of only 0.87. In the row 2nd row and 3rd column we have value 0.12 which basically means about 12% readings of the class sitting is misclassified as standing.<br>
<br>
It is not a surprise as in the t-sne plot (plot-5) we saw that 'sitting' and 'Standing' readings are overlapping.

For detailed code of all the ML models check the [HAR_PREDICTION_MODELS Notebook](https://github.com/srvds/Human-Activity-Recognition/blob/master/HAR_PREDICTION_MODELS.ipynb)

#### LSTM Model

keras with tensorflow backend is used.

**LSTM model 1**

``` python
timesteps = len(X_train[0])
input_dim = len(X_train[0][0])
n_classes = _count_classes(Y_train)

print(timesteps)
print(input_dim)
print(len(X_train))
```
```
128
9
7352
```
For detailed code of this section you can always check the [HAR_LSTM Notebook](https://github.com/srvds/Human-Activity-Recognition/blob/master/HAR_LSTM.ipynb)

