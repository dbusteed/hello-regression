using Statistics
using CSV   # using Pkg; Pkg.add("CSV")

#
# STEP 1
#
#  read the data
#
cars = CSV.read("cars.csv")
cars = convert(Matrix, cars)


#
# STEP 2
#
#   split data into "in-sample data" and "out-of-sample" data
#   (also known as "train" and "test" sets)
#
split = Int64(floor(size(cars)[1] * .75))
in_sample_data = cars[1:split, :]
out_sample_data = cars[(split+1):size(cars)[1], :]


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
y = in_sample_data[:, 1]

X = in_sample_data[:, 2:size(in_sample_data)[2]]
X = [ones(size(in_sample_data)[1]) X]


#
# STEP 4
# 
#   calculate the vector of model parameters according to 
#   the OLS proof, which selects the parameters which
#   minimizes the sum of squared residuals
#
β = inv(transpose(X) * X) * transpose(X) * y

println("Model parameters: ")
println("\t", β, "\n")


#
# STEP 5
#
#   create a function that applies the model parameters (beta)
#   to a any set of matching model predictors
#
#   β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ
#
function predict_mlr(β, xs)
    β[1] + sum(β[2:length(β)] .* xs)
end


#
# STEP 6
#
#   use the above function to predict the MPG (our response
#   variable) for each data point within our "out-of-sample"
#   dataset
#
predictions = Float64[]
for i in 1:size(out_sample_data)[1]
    xs = out_sample_data[i, 2:size(out_sample_data)[2]]
    pred = predict_mlr(β, xs)
    push!(predictions, pred)
end


#
# STEP 7
#
#   grab the actual values for our response variable from the
#   testing dataset, and use them (and our predictions) to compute
#   the R-Squared for our model
# 
#   R² = 1 - (SSE/SST), where SSE is the Sum of Squared Regression Error,
#   and SST is the Sum of Squared Total Error
#
actuals = out_sample_data[:, 1]

SSE = sum((actuals .- predictions) .^2)
SST = sum((actuals .- Statistics.mean(actuals)) .^2)

R² = (1 - (SSE/SST))

println("Out-of-Sample R-Squared: ")
println("\t", R², "\n")