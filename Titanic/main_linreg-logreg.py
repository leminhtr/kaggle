import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model import LinearRegression as linreg
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.cross_validation import KFold
from sklearn.cross_validation import *
from sklearn import  cross_validation

titanic=pd.read_csv("train.csv")

#print(titanic.describe())
#print(titanic.head(5))

# ------------------- DATA CORRECTION --------------------------------

# 1) Fill missing Age data with median
titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())

# 2) Convert Sex string with 0 or 1
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0 #convert 0 for men
titanic.loc[titanic["Sex"] =="female", "Sex"]=1 #convert 1 for women

# 3) Fill missing Embarked data with most common char
print(pd.value_counts(titanic["Embarked"].values, sort=False))
     # "S" is most common char -> chosen as default for missing values
titanic["Embarked"]=titanic["Embarked"].fillna("S")

#4) Replace Embarked char with numeric code
#titanic.loc[titanic["Embarked"]=="S", "Embarked"]=0 # 'S' -> 0
#titanic.loc[titanic["Embarked"]=="C", "Embarked"]=1 # 'C' -> 1
titanic.loc[titanic["Embarked"]=="S", "Embarked"]=0
titanic.loc[titanic["Embarked"]=="C", "Embarked"]=1
titanic.loc[titanic["Embarked"]=="Q", "Embarked"]=2 # 'Q' -> 2


# input column used for predictions :
predictors=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize the algorithm
algo_linreg = linreg()

# Generate cross-validation folds with random splits
# return rows indices for corresponding train and set
kf =KFold(titanic.shape[0], n_folds=3, random_state=1)

# Make the predictions
predictions =[]
for train, test in kf:
    # Which predictors used on train fold
    train_predictors = (titanic[predictors].iloc[train,:])
    # Target/goal used to train the algo
    train_target= titanic["Survived"].iloc[train]

    # Train the algo with the predictors and target
    # .fit(x input, y output)
    algo_linreg.fit(train_predictors, train_target)
    # Make predictions with the trained algo on test fold
    test_predictions = algo_linreg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)

# The predictions are in 3 Numpy arrays
# So we concatenate the arrays on axis 0 (bc only 1 axis)
predictions=np.concatenate(predictions, axis=0)

predictions[predictions> .5]=1
predictions[predictions<= .5]=0

print(predictions)

print(sum(predictions==titanic["Survived"]))
accuracy= sum(predictions==titanic["Survived"])/len(predictions)
print(accuracy) # = 0.783


#------------------- Logistic Regression method ---------------------

# Initialize the algo
algo_logreg = logreg(random_state=1)

# Compute accuracy score for all cross-V folds;
# cross_val_score(algo, predictors, target, cross-validation fold)
scores = cross_validation.cross_val_score(algo_logreg, titanic[predictors], titanic["Survived"], cv=3)
# Mean of the scores for each folds (3 folds)
print(scores.mean())


#----------------------------------- Log Reg. with test set ---------------------

titanic_test = pd.read_csv("test.csv")

# I) Clean data
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

# II) Test algo on data

# Initialize the algo
algo_logreg_test=logreg(random_state=1)

# Train algo on using all training data
algo_logreg_test.fit(titanic[predictors], titanic["Survived"])

# Make predictions with algo on data
predictions=algo_logreg_test.predict(titanic_test[predictors])

# Generate new dataset for kaggle submission
submission= pd.DataFrame({
    "PassengerId" : titanic_test["PassengerId"],
    "Survived": predictions
})

submission.to_csv("kaggle.csv", index=False)

