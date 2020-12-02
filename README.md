# Humidity-Predictor

This is a simple regression-based project that predicts the Absolute and Relative humidity present in the air given variables such as Temperature, Benzene Concentration, Nitrogen Dioxide Concentration etc.

Two seperate scripts perform this function while also comparing several models on the bases of R Squared and Adjusted R Squared.

A) The output of the Absolute Humidity Prediction script is:

|Number|       Regression Models       |R Squared  |Adjusted R Squared| Single Predictions |
|:----:|:-----------------------------:|:---------:|:----------------:|:------------------:|
|  1   |        Linear Regression      |   0.81    |       0.81       |        2.20        |
|  2   |     Polynomial Regression     |   0.90    |       0.90       |        3.49        |
|  3   |   Support Vector Regression   |   0.93    |       0.93       |        0.91        |  
|  4   |   Decision Tree Regression    |   0.82    |       0.82       |        1.92        |
|  5   |   Random Forrest Regression   |   0.92    |       0.92       |        1.37        |
|  6   | Gradient Boosting Regression  |   0.87    |       0.87       |        1.69        |


B) The output of the Relative Humidity Prediction script is:

|Number|       Regression Models       |R Squared  |Adjusted R Squared| Single Predictions |
|:----:|:-----------------------------:|:---------:|:----------------:|:------------------:|
|  1   |        Linear Regression      |   0.76    |       0.76       |      104.32        |
|  2   |     Polynomial Regression     |   0.88    |       0.88       |       67.45        |
|  3   |   Decision Tree Regression    |   0.75    |       0.75       |       78.80        |
|  4   |   Random Forrest Regression   |   0.89    |       0.89       |       70.07        |
|  5   | Gradient Boosting Regression  |   0.82    |       0.82       |       70.18        |

Notes:

1) These scripts were made with the help of Numpy, Pandas and Scikit-Learn. 
2) There is no noticable difference between R Squared and adjusted R Squared in the tables above. This is due to small differences getting rounded off. Please see the output.csv files for exact values.
3) The .jpg file provides more information about the variables found in the dataset.
4) The Single predictions are made on a list containing features with randomized but realistic values.
5) Support Vector Regression was excluded from the Relative Humidity prediction script due to it resulting in negative R Squared.

Dataset Link: https://archive.ics.uci.edu/ml/datasets/Air+Quality
