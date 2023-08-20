# Exercise 1: Introduction to Machine Learning

## Setup

Make sure that you are inside of the `intro_ml` folder by using the `cd` command to change directories if needed.

Run the setup script to create the environment for this exercise and download the dataset.
```bash
source setup.sh
```

Launch a jupyter environment

```
jupyter lab
```

...and continue with the instructions in the notebook `exercise.ipynb`.

## Exercise

### Part A: The Linear Classifier
We will implement a basic linear classifier from scratch and train it to predict the cell cycle on a flow cytometry dataset.

You will learn
- How to prepare a dataset for training, including
    - Checking for class imbalance (Task 1.1)
    - Correcting class imbalance (Task 1.2)
    - Converting categorical data to one-hot encoding (Task 1.3)
- The basic math behind a linear classifier
- How to evaluate model performance (Tasks 2.1 - 2.3)

### Part B: Random Forest Classifier
We will learn about Random Forest Classifiers and use `scikit-learn` to train one on our dataset.

You will learn
- How to use `scikit-learn`'s model objects
- How to perform a hyperparameter search to optimize model performance (Tasks 3.1 - 4.1)

### Part C: Feature Engineering
We will explore image filters and see if they can improve the performance of either our linear or random forest classifer.

You will learn
- How to use `scikit-image`'s filter modules (Task 5.1)
- How to reuse what we have done so far to build a new dataset on filtered images and train your own model on the new dataset (Task 5.2)