agenda for 7/26

##### recap of last time

- we tried to fit the first spaced repetition model, we used only 2 features: history_seen, history_correct
- we preprocessed the data to get into form to allow pytorch to process it
- we got stuck, due to some errors in model fitting - due to a problem we didn't recognize

- we actually later found out that we were ran into the exploding gradients problem.
- to solve this, we simply need to scale the data.

##### today

- we'll fit a model
- we'll evaluate how good the initial model is
- we'll have a menu of options for what to do next

**menu**:

- feature engineering
- comparison against other baseline models
-- simple naive baseline: to always predict 50% recall
-- leitner system, pimsleur system - these are models that have been used for a long time.
- how can we improve the model training process. are we training for long enough, are we choosing the correct hyperparameters (learning rate, regularization strength (how big of a penalty term should we have))
- so far we're only working with 1% of the data (12k rows out of 1.2M) eventually we will want to train the model on all the data. 


agenda for 7/19

##### recap of last time

- conceptual review and understanding the inner workings of the half-life model

##### today

- if the model development pipeline consists of the following steps
-- **data preprocessing and validation** (our data is clean and makes sense)
-- **conceptual understanding of the modeling task** (predicting a half-life for the memory of a word)
-- **feature engineering** (we're not doing it yet - we're simply taking the features that are already there)
-- **model specification** (writing the model class in pytorch)
-- **finding good parameters** for the model/ optimization
-- **model evaluation** (judging how good the model is)

we will quickly move through the entire process first to make sure we have the right ideas at each step.

then we will improve our work at each step in subsequent passes.

in other words, we are agile.

- we will try to fit our model and we need to evaluate it against a baseline model
- **baseline model**:
-- naive model: always predict p_recall to be 50% (in other words: shrug, complete ignorance)
-- slightly less naive model: leitner system and pimsleur method

compare leitner system and pimsleur method to our model.

1. we will take the error between observed p_recall and leitner/pimsleur
2. we will take the error between observed p_recall and our model

and if 1. is greater than 2. then our model is an improvement over the baseline.

**a few more words about baseline**:

classification/regression ML tasks have naive baseline models as well.

**classification**

suppose we are predicting who will default on a loan (not pay it back) using logistic regression.

what is the naive baseline model in this scenario?

more context: suppose the class balance (ratio between those who pay and those who don't) is 2:1

if you had no information about a specific customer and their loan, what would be a good baseline prediction?

random guessing = guess that they will pay with 67% probability

in other words: if your model is to always predict "will pay back" your accuracy is 67%

so if your logistic regression CANNOT beat accuracy 67% then it is USELESS (no better than having a constant prediction).

**regression**

you are predicting home values for a real estate website.

you will use a regression model.

what is the naive baseline model?

if you always predict the mean you will have an R-squared of 0.

if your regression model has an R-squared close to 0, then it is useless because you can always just predict the mean and perform just as well.

**classification - of cancer in a medical image**

we could use the base rate of non-cancer vs cancer. suppose it is 10:1

so if you always predict non-cancer you will have 91% accuracy.

but is this really the best baseline model?

**how do we decide if there's cancer in an image when we don't have an ML model?**

traditionally, human doctors look at images and decide if there's cancer.

what is the classification accuracy of doctors?

if the accuracy of human doctors is 96%

**we need to beat 96% accuracy**

**question**:

1. model 1: always predict non-cancer --> accuracy: 91%
2. model 2: use trained human --> accuracy: 96%

definition of accuracy:

correct predictions / total predictions

correct predictions = {true positives, true negatives}

true positive: predicted cancer, truly cancer
true negative: predicted non-cancer, truly non-cancer

example data:

100 images, 9 are cancerous, 91 are not

model 1 accuracy: (0 + 91)/100, has 9 false negatives
model 2 accuracy: (5 + 91)/100, they have 4 false negatives

agenda for 7/12

##### recap of last time

- feature engineering - vectorizing parts of speech for practiced words //
-- expecting 92 unique tokens, but saw many more
- talked about the model
-- introduced pytorch
-- object oriented programming (since we need write a custom model class)

##### today

- circle back on feature engineering - finish it (for now)
- focus on the model
- review its components (conceptually)
- see how its implemented
- possibly, we will try to optimize the function with something like gradient descent.


agenda for 7/5

- recap of last time 
- feature engineering - one hot encoding lexeme attributes
- check for more feature engineering opportunities - using kaggle and duolingo paper/code as references
- set up the learning objective 

agenda for 6/28

##### overview of the objective

well timed reviews help people learn better - according to well established results in the cog sci of learning (and according to common sense). what does well-timed mean? enough time such that the answer to the review question is not super obvious - but not so much time that the answer is completely out of memory.
 
once you've been exposed to a vocab word in a language you are studying (in this scenario, you are not studying immersively), the clock starts on the decay of your memory of the word. if you review the word immediately after learning it, you will almost certainly remember what it means. but if you wait for a long time - say, a year - you will probably not be able to recall its meaning. you can see that in both cases the review is not timed well. it'd be better to review the word when you have a decent shot at remembering (the probability of recall is not too low) but also have to work for it a little bit (..and not too high). so the `next time you review` should be decided as a function of the `probability of recall`. 

but what is the `probability of recall` a function of, and what kind of function is it?

it is thought to decay exponentially in time.

quick pass over the duolingo paper

notes:

they use the half-life model to represent decay in probability of recall

```python

p = 2**(-1*(delta/half_life))

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

    squared_difference = (df.loc[i,'y'] - df.loc[i,'y_hat'])**2
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

