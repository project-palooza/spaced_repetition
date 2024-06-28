### building a spaced repetition model with duolingo data

##### agenda for 6/28/2024

first session

- overview of the objective

well timed reviews help people learn better
what does well-timed mean?
enough time such that the answer to the review question is not super obvious
but not so much time that the answer is completely out of memory

`next_review(x_user,x_item)`

`p_recall(x_user,x_item)`

next_review should be something like, when `p_recall` is predicted to be .5

so we're going to build (learn) these functions

- quick pass over the duolingo paper

notes:

they use the half-life model to represent decay in probability of recall

```python

p = 2**(delta/half_life)

```

introducing `x` and `theta`

`x` is a "feature vector" representing a students previous exposure to a word 

e.g. `x` could be the history of my reviews of the spanish word gato `x` = (# of times reviewed, # of times correctly recalled) = (12,8)

`theta` is a vector of weights representing the importance of each value in `x` to predicting the probability of recall

e.g. suppose # of times reviewed is twice as important as # of times correctly recalled `theta` = (theta_1,theta_2) = (.002, .001)

`x*theta` = .002*12 + .001*8 = .032

`predicted half life = 2**(x*theta)`

so in our particular case

`predicted half life = 2**.032` which is approximately 1

predicted probability of recall is `2**-(1/1) = 1/2`

two big tasks

- find good features `x`
- learn good feature weights `theta`

to find good feature weights, we will use a machine learning approach

in general ML works like this

we want to learn some function parameters (in our case we want to know what `theta` should be)

we set up an objective

minimize the error between predicted and observed data

what is the observed data in our case?

we know the recall rate (correct review rate) we have a historical correct review rate, and we call this observed p (the probability of recall)

compare observed p to predicted p - and quantify how wrong the prediction is

observed p = .9
predicted p = .5

`observed p - predicted p = .9 - .5 = .4`

kick off a iterative learning process - in ML this is called gradient descent

`error()`

in ML - depending on the type of the data you are trying to predict, and the model that you use there are different to quantify the error of your predictions.

we are trying to predict `y` and `y` is continuous

and suppose we use linear regression to build a function `f(x) = W*x = y_hat`

and we want to compare `y` to `y_hat` to see how wrong we are

a popular loss function (error function) is mean squared error

```python

mse = 0

for i in range(df.shape[0]):

    squared_difference = (df['y'] - df['y_hat'])**2
    mse += squared_difference

```

- formulate next steps

- EDA
- (possibly) preprocessing
- EDA
- (possibly) feature engineering (creating new columns based on existing columns)
- EDA
- model fitting
- model evaluation

