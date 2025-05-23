import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/prade/OneDrive/Desktop/Advertising.csv")
print('DATA is---------',df.head())

# Everything present EXCEPT the sales column
X = df.drop('sales',axis=1)
y = df['sales']

from sklearn.preprocessing import PolynomialFeatures
polynomial_converter = PolynomialFeatures(degree=2,include_bias=False)
#include_bias=False because we din't want to put any biase 

poly_features = polynomial_converter.fit(X)
poly_features = polynomial_converter.transform(X)

# OR, we can do it in a single line too........
# Converter "fits" to data, in this case, reads in every X column
# Then it "transforms" and ouputs the new polynomial data
# poly_features = polynomial_converter.fit_transform(X)

print('The shape of the new data is: ',poly_features.shape)
print('The shape of the original data was : ',X.shape)


#Now we are going to train our model on this new data(poly_features)...
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(X_train,y_train)
test_predictions = model.predict(X_test)

#To evaluate the model..
from sklearn.metrics import mean_absolute_error,mean_squared_error
MAE = mean_absolute_error(y_test,test_predictions)
MSE = mean_squared_error(y_test,test_predictions)
RMSE = np.sqrt(MSE)

print('mean_absolute_error: ',MAE)
     #result:-  0.48967980448037096
print('mean_squared_error: ',MSE)
     #result:- 0.44175055104035904
print('root_mean_squared_error: ',RMSE)
     #result:- 0.6646431757269152

print('Whereas the average sales is: ',df['sales'].mean())

     # we can easily see these results are much better than the previous model...

#-------------------------------------------------------------------------------------------------------
# Now try to adjust the model, to increse the it's efficiency if possible

# TRAINING ERROR PER DEGREE
train_rmse_errors = []
# TEST ERROR PER DEGREE
test_rmse_errors = []

for d in range(1,10):
    
    # CREATE POLY DATA SET FOR DEGREE "d"
    polynomial_converter = PolynomialFeatures(degree=d,include_bias=False)
    poly_features = polynomial_converter.fit_transform(X)
    
    # SPLIT THIS NEW POLY DATA SET
    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)
    
    # TRAIN ON THIS NEW POLY SET
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train,y_train)
    
    # PREDICT ON BOTH TRAIN AND TEST
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate Errors
    
    # Errors on Train Set
    train_RMSE = np.sqrt(mean_squared_error(y_train,train_pred))
    
    # Errors on Test Set
    test_RMSE = np.sqrt(mean_squared_error(y_test,test_pred))

    # Append errors to lists for plotting later
    
   
    train_rmse_errors.append(train_RMSE)
    test_rmse_errors.append(test_RMSE)

plt.plot(range(1,6),train_rmse_errors[:5],label='TRAIN')
plt.plot(range(1,6),test_rmse_errors[:5],label='TEST')
plt.xlabel("Polynomial Complexity")
plt.ylabel("RMSE")
plt.legend()
plt.show()

# Based on our chart, could have also been degree=4, but 
# it is better to be on the safe side of complexity
final_poly_converter = PolynomialFeatures(degree=3,include_bias=False)

final_model = LinearRegression()
final_model.fit(final_poly_converter.fit_transform(X),y)

# Let's ckeck the final model........
campaign = [[149,22,12]]
modified_campaign=final_poly_converter.fit_transform(campaign)
estimated_sales= final_model.predict(modified_campaign)

print('Estimated sales on any random data:', campaign, 'is:',estimated_sales)


# print('HELLOW WORLD.........')
# Saving the model and converter....
from joblib import dump
# Save the polynomial converter
dump(final_poly_converter, 'sales_poly_converter.joblib')

# Save the trained model
dump(final_model, 'sales_poly_model.joblib')

