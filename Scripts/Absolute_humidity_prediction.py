# Predicting Air Humidity and comparing different Regression models

# Initiating starting time 
import time
start = time.time()

# Importing libraries and dataset
import numpy as np
import pandas as pd
import statsmodels.api as sm
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

dataset = pd.read_csv('C:/Users/Sukant Sidnhwani/Desktop/Python/Projects/Absolute Humidity Prediction/AirQualityUCI.csv')

X = dataset.iloc[:, 2 : -2].values # Excluding Relative Humidity from dataset because otherwise it'd be too easy for the model
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)

# Check for missing values 
dataset.isnull().any()
dataset.isnull().sum()

# Taking care of missing values in X
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 0:])
X[:, 0:] = imputer.fit_transform(X[:, 0:])
# imputer = imputer.fit(X[:, 0:].reshape(-1, 1)) : Uncomment in case it gives an error 

# Rechecking [X]
a = pd.DataFrame(X)
a.isnull().any()

# Taking care of missing values in y
imputer2 = SimpleImputer(missing_values = np.nan, strategy = 'median')
imputer2 = imputer2.fit(y)
y = imputer.fit_transform(y)
# imputer2 = imputer2.fit(y.reshape(-1, 1)) : Uncomment in case it gives an error 

# Rechecking [y]
b = pd.DataFrame(y)
b.isnull().any()

# Spliting the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.3)

# Linear Regression using Scikit Learn Module
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

one = r2_score(y_test, y_pred)

# Linear Regression using statsmodel
X_with_constant = sm.add_constant(X_train)
Xt_with_constant = sm.add_constant(X_test)
ols_reg = sm.OLS(y_train, X_with_constant)
ols_reg = ols_reg.fit()
y_pred2 = ols_reg.predict(Xt_with_constant)

two = r2_score(y_test, y_pred2)

# Polynomial Regression
pol = PolynomialFeatures(degree = 3)
x_transformed = pol.fit_transform(X_train)
xt_transformed = pol.fit_transform(X_test)
poly_reg = LinearRegression()
poly_reg.fit(x_transformed, y_train)
y_pred3 = poly_reg.predict(xt_transformed)

three = r2_score(y_test, y_pred3)

# Support Vector Regression 
sc_X = StandardScaler()
sc_y = StandardScaler()
sc_Xt = StandardScaler()
sc_yt = StandardScaler()
X = sc_X.fit_transform(X_train)
Xt = sc_Xt.fit_transform(X_test)
y = sc_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
yt = sc_yt.fit_transform(y_test.reshape(-1, 1)).reshape(-1)
sv_reg = SVR(kernel = 'rbf')
sv_reg = sv_reg.fit(X, y)
y_pred4 = sv_reg.predict(Xt)
# y_pred4 = sc_Xt.inverse_transform(y_pred4)

four = r2_score(yt, y_pred4)

# Decision Tree Regression
dt_reg = DecisionTreeRegressor()
dt_reg = dt_reg.fit(X_train, y_train)
y_pred5 = dt_reg.predict(X_test)

five = r2_score(y_test, y_pred5)

# Random Forrest Regression 
rf_reg = RandomForestRegressor(n_estimators = 100)
y_train = y_train.reshape(-1, 1).reshape(-1)
rf_reg = rf_reg.fit(X_train, y_train)
y_pred6 = rf_reg.predict(X_test)

six = r2_score(y_test, y_pred6)

# Gradient Boosting Regression
xgbr = XGBRegressor(learning_rate = 0.09)
xgbr = xgbr.fit(X_train, y_train)
y_pred7 = xgbr.predict(X_test)

seven = r2_score(y_test, y_pred7)

# Evaluating all R squares in a list and turning it into a dataframe 
list1 = [one, two, three, four, five, six, seven]

# Converting these R squares into adjusted R square
list2 = []
for i in list1:
    vals = adjusted_r_squared = 1 - (1 - i) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
    list2.append(vals)

# Predicting a value with different regressors 

# Creating demo values 
demo = [3.4, 1690, 104, 10.3, 640, 245, 681, 87, 1856, 1004, 17.39]   #, 49.13]
demo = pd.DataFrame(demo)
demo = demo.transpose()
demo = demo.to_numpy()

# Additional transformations 
demo_with_constant = sm.add_constant(demo, has_constant = 'add') # Adding constant to demo to fit OLS regression
demo_poly = pol.fit_transform(demo) # Transforming demo to 3rd degree to fit Polynomial regression
sc_z = StandardScaler() # SVR feature scaling 
demo_svr = sc_z.fit_transform(demo) # SVR feature scaling

# Applying different regressors 
a = lin_reg.predict(demo)
b = ols_reg.predict(demo_with_constant)
c = poly_reg.predict(demo_poly)
d = sv_reg.predict(demo)
d = sc_y.inverse_transform(d)
e = dt_reg.predict(demo)
f = rf_reg.predict(demo)
g = xgbr.predict(demo)

# Making a list of demo predictions
list3 = [a, b, c, d, e, f, g]
list3 = np.array(list3, dtype = 'object')

# Initiating ending time
end = time.time()
print()
print(f"This program executes in {round((end - start), 2)} seconds.")
print()

# Concatenating R squares and Demo prediction lists into a dataframe
df3 = pd.DataFrame()
df3['Models'] = ['Linear Regression (Sklearn)', 
                 'Linear Regression (Statsmodels)', 
                 'Polynomial Regression', 
                 'Support Vector Regression', 
                 'Decision Tree Regression', 
                 'Random Forrest Regression',
                 'Gradient Boosting Regression']
df3['R Squared'] = list1
df3['Adjusted R Squared'] = list2
df3['Demo Predictions'] = list3

# Saving the output as a .csv file. Already done so commented out.
# df3.to_csv('AH_Output.csv')
