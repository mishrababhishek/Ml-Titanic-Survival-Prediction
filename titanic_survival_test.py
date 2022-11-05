
#imports
import random
import pickle 
import columnar
import pandas as pd
from sklearn.linear_model import LogisticRegression


# Import and Analysis Data
titanic_test = pd.read_csv("data/test.csv")
test_names = titanic_test["Name"]
titanic_test
test_names


# Removeing Null and Redundant Row and column
titanic_test = titanic_test.drop(["Cabin", "Ticket", "Name", "PassengerId"], axis=1)
titanic_test = titanic_test.dropna()


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
titanic_test["Sex"] = titanic_test["Sex"].apply(sex_conveter)
titanic_test["Embarked"] = titanic_test["Embarked"].apply(embarked_conveter)
titanic_test["Age"] = titanic_test["Age"].apply(int)

titanic_test


# Load Trained Model
with open("titanic.pkl", "rb") as f:
    global lr_model
    lr_model :LogisticRegression = pickle.load(f)


# Create Testing Data
train_x = titanic_test.iloc[:, ].values
train_x


# Predict and save Survivors to file
result = lr_model.predict(train_x)
with open("Survivors.txt", "wb") as f:
    data = []
    for index, r in enumerate(result):
        d = [test_names[index], "Survived" if r == 1 else "Not Survived"]
        data.append(d)
    table = columnar.columnar(data, headers=("Names", "Status"))
    f.write(table.encode())
    f.close()



