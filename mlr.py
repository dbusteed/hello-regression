import pandas as pd   # pip install pandas
import numpy as np    # pip install numpy
import math

#
# STEP 1
#
#  read the data
#
cars = pd.read_csv('cars.csv')


#
# STEP 2
#
#   split data into "in-sample data" and "out-of-sample" data
#   (also known as "train" and "test" sets)
#
split = math.floor(len(cars) * .75)
in_sample_data = cars.loc[:(split-1)]
out_sample_data = cars.loc[split:]


#
# STEP 3
#
#   create matricies for our "y" and "X" variables. in this
#   case, the "y" or response variable is the MPG rating of
#   a given car. the "X" variables or predictors are the other
#   features of the car, that we will use to predict the MPG.
#
#   notice how we add a column of ones to the beginning of the X
#   matrix, so that the model intercept can be included in 
#   the prediction
#
y = in_sample_data.iloc[:, 0]
y = np.matrix(y).T

X = in_sample_data.iloc[:, 1:]
X = np.matrix(X)
X = np.append(np.ones(24).reshape(24,1), X, axis=1)


#
# STEP 4
# 
#   calculate the vector of model parameters according to 
#   the OLS proof, which selects the parameters which
#   minimizes the sum of squared residuals
#
#   note: in R, `t()` is a matrix transpose,
#   and `solve()` is a matrix inverse
#
B = (X.T * X).I * X.T * y

print("Model parameters:")
print(f"\t{B.T}\n")


#
# STEP 5
#
#   create a function that applies the model parameters (beta)
#   to a any set of matching model predictors
#
#   B0 + B1*X1 + B2*X2 + ... + Bk*Xk
#
def predict_mlr(beta, xs):
    return beta[0] + sum(np.array(xs) * np.array(beta[1:]))


#
# STEP 6
#
#   use the above function to predict the MPG (our response
#   variable) for each data point within our "out-of-sample"
#   dataset
#
predictions = []
for i in range(len(out_sample_data)):
    pred = predict_mlr(np.array(B).reshape(-1,), out_sample_data.iloc[i, 1:])
    predictions.append(pred)


#
# STEP 7
#
#   grab the actual values for our response variable from the
#   testing dataset, and use them (and our predictions) to compute
#   the R-Squared for our model
# 
#   R^2 = 1 - (SSE/SST), where SSE is the Sum of Squared Regression Error,
#   and SST is the Sum of Squared Total Error
#
actuals = out_sample_data.iloc[:, 0]

actuals = np.array(actuals)
predictions = np.array(predictions)

SSE = sum((actuals - predictions)**2)
SST = sum((actuals - np.mean(actuals))**2)

R2 = (1 - (SSE/SST))

print("Out-of-Sample R-Squared: ")
print(f"\t{R2}")