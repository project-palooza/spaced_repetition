

agenda for 6/28/2024

##### overview of the objective

well timed reviews help people learn better - according to well established results in the cog sci of learning (and according to common sense). what does well-timed mean? enough time such that the answer to the review question is not super obvious - but not so much time that the answer is completely out of memory.
 
once you've been exposed to a vocab word in a language you are studying (in this scenario, you are not studying immersively), the clock starts on the decay of your memory of the word. if you review the word immediately after learning it, you will almost certainly remember what it means. but if you wait for a long time - say, a year - you will probably not be able to recall its meaning. you can see that in both cases the review is not timed well. it'd be better to review the word when you have a decent shot at remembering (the probability of recall is not too low) but also have to work for it a little bit (..and not too high). so the `next time you review` should be decided as a function of the `probability of recall`. 

but what is the `probability of recall` a function of, and what kind of function is it?

it is thought to decay exponentially in time.

quick pass over the duolingo paper

notes:

they use the half-life model to represent decay in probability of recall

```python

p = 2**(delta/half_life)

```

introducing `x` and `theta`

`x` is a "feature vector" representing a students previous exposure to a word 

e.g. `x` could be the history of my reviews of the spanish word gato `x` = (# of times reviewed, # of times correctly recalled) = (12,8)

`theta` is a vector of weights representing the importance of each value in `x` to predicting the probability of recall

e.g. imagine if # of times reviewed is twice as important as # of times correctly recalled `theta` = (theta_1,theta_2) = (.002, .001)

we then apply the weights `theta` to features `x` with a dot product.

so in our particular case

`x*theta` = .002*12 + .001*8 = .032

`predicted half life = 2**.032` which is approximately 1

and predicted probability of recall is `2**-(1/1) = 1/2`

two tasks ahead

- find good features `x`
- learn good feature weights `theta`

to find good feature weights, we will use a machine learning approach

in general ML works like this

we want to learn some function parameters (in our case we want to know what `theta` should be)

we set up an objective to minimize the error between predicted and observed data then kick off a iterative learning process to obtain the best parameters we can in the context of the model we chose - in ML this is called gradient descent.


depending on the type of the data you are trying to predict, and the model that you use, there are different ways to quantify the error of your predictions. for example, if we are trying to predict `y` as a function of `x` and `y` is continuous, we might use linear regression `f(x) = W*x = y_hat` (no intercept, for simplicity).

we will want to compare `y` to `y_hat` to see how wrong we are

a popular loss function (error function) is mean squared error

```python

mse = 0

for i in range(df.shape[0]):

    squared_difference = (df['y'] - df['y_hat'])**2
    mse += squared_difference

```

next steps

- EDA
- (possibly) preprocessing
- EDA
- (possibly) feature engineering (creating new columns based on existing columns)
- EDA
- model fitting
- model evaluation

