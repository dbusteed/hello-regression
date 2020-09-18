#
# STEP 1
#
#  read the data
#
cars <- read.csv("cars.csv")


#
# STEP 2
#
#   split data into "in-sample data" and "out-of-sample" data
#   (also known as "train" and "test" sets)
#
split <- floor(nrow(mtcars) * .75)
in_sample_data  <- cars[1:split, ]
out_sample_data <- cars[(split+1):nrow(cars), ]


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
y <- in_sample_data[1]
y <- t(matrix(unlist(y), ncol=nrow(y), byrow=TRUE))

X <- in_sample_data[2:ncol(in_sample_data)]
X <- t(matrix(unlist(X), ncol=nrow(X), byrow=TRUE))
X <- cbind(1, X)


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
beta <- solve((t(X) %*% X)) %*% t(X) %*% y

cat("Model parameters: \n")
cat("\t", t(beta), "\n\n")


#
# STEP 5
#
#   create a function that applies the model parameters (beta)
#   to a any set of matching model predictors
#
#   B0 + B1*X1 + B2*X2 + ... + Bk*Xk
#
predict.mlr <- function(beta, xs) {
  beta[1] + sum(beta[2:length(beta)] * xs)
}


#
# STEP 6
#
#   use the above function to predict the MPG (our response
#   variable) for each data point within our "out-of-sample"
#   dataset
#
predictions <- c()
for(i in 1:nrow(out_sample_data)) {
  xs <- unlist(out_sample_data[i, 2:ncol(out_sample_data)], use.names=FALSE)
  pred <- predict.mlr(beta, xs)
  predictions <- c(predictions, pred)
}


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
actuals <- out_sample_data[, 1]

SSE <- sum((actuals - predictions)^2)
SST <- sum((actuals - mean(actuals))^2)

R2 <- (1 - (SSE/SST))

cat("Out-of-Sample R-Squared: \n")
cat("\t", R2, "\n\n")