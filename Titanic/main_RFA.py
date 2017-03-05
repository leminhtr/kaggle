from sklearn import cross_validation
from sklearn.cross_validation import *
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

titanic= pd.read_csv("train.csv")


# ------------------- I) DATA CORRECTION -----------------------------

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

#--------------------------------------------------------------------

# input column used for predictions :
predictors=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize the algorithm
algo_RFA= RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

# Generate cross-validation folds with random splits
# return rows indices for corresponding train and set
kf= KFold(titanic.shape[0], random_state=1, n_folds=3)

scores=cross_validation.cross_val_score(algo_RFA, titanic[predictors], titanic["Survived"], cv=kf)

print(scores.mean()) #0.7856