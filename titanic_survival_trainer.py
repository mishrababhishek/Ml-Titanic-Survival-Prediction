
#imports
import random
import pickle 
import pandas as pd
from sklearn.linear_model import LogisticRegression


# Import and Analysis Data
titanic_train = pd.read_csv("data/train.csv")
titanic_train


# Removeing Null and Redundant Row and column
titanic_train = titanic_train.drop(["Cabin", "Ticket", "Name", "PassengerId"], axis=1)
titanic_train = titanic_train.dropna()

titanic_train.isna().sum()


# Conveter Functions
def sex_conveter(value :str) -> int:
    if value == "male":
        return 0
    elif value == "female":
        return 1
    else:
        return -1

def embarked_conveter(value :str) -> int:
    if value == "S":
        return 0
    elif value == "C":
        return 1
    elif value == "Q":
        return 2
    else:
        return -1


# Transforming Labels
titanic_train["Sex"] = titanic_train["Sex"].apply(sex_conveter)
titanic_train["Embarked"] = titanic_train["Embarked"].apply(embarked_conveter)
titanic_train["Age"] = titanic_train["Age"].apply(int)



titanic_train


# Creating Train and Test Set

train_x = titanic_train.iloc[: , 1:8].values
train_y = titanic_train.iloc[: , 0].values
temp = list(zip(train_x, train_y))
random.shuffle(temp)
train_x, train_y = zip(*temp)



# Creating and Training Model
lr_model = LogisticRegression()
lr_model.max_iter = 150
lr_model.fit(train_x, train_y)
lr_model.score(train_x, train_y)


# Save Model
with open("titanic.pkl", "wb") as f:
    pickle.dump(lr_model, f)
    f.close()


