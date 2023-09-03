import numpy as np



## Input vector
vector1 = np.array([2, 1.5])

## Setting the weight for each node or layer?
weight1 = np.array([1.45, -.66])
##weight2 = np.array([2.17, 0.23])

### adding bias
bias = np.array([0])


## calculating the dot_product
def dot_product(vector, weight):

    ## Calculating the dot-product of vector1 and weight1
    ##1st multiply all corespomding indexes
    index0 = vector[0] * weight[0]
    index1 = vector[1] * weight[1]

    ## add all products together for dot-product
    return index0 + index1

## getting a non liniear function sigmoid
def sigmoid(x):
    return 1/(1+ np.exp(-x))


## calculating a prediction
def prediction(vector, weight, bias):
    layer1 = dot_product(vector, weight)
    layer2 = sigmoid(layer1)

    return layer2


## calculate the prediction error via mean squared error



target = 0
error = np.square(prediction(vector1, weight1, bias) - target)


dydx = 2*(prediction(vector1, weight1, bias) - target)
weight1 = weight1 - dydx

prediction1 = prediction(vector1, weight1, bias)
error = (prediction1 - target) ** 2

## Output
## We have a look at which one is the biggest, because the biggest ummber implies a higher match
pred = prediction(vector1, weight1, bias)

print(f'Prediction: {pred}; Error: {error};')