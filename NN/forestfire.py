#PREDICT THE BURNED AREA OF FOREST FIRES WITH NEURAL NETWORKS
#importing all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#for neural network algorithm importing libraries and functions
from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda

#For splitting dataset into test and train 
from sklearn.model_selection import train_test_split

#Loading dataset forestfires.csv and checking its information 
forest = pd.read_csv("forestfires.csv")
forest.head()

forest.info()

#target variable size_category information and counts
forest.size_category.describe()
forest.size_category.value_counts()

#EDA
forest.isnull().sum()

#  small as 0 and large as 1

forest.loc[forest.size_category=="small","size_category"] = 0
forest.loc[forest.size_category=="large","size_category"] = 1

#Preprocessing Data by proving labels to categorical data
string_columns=["month","day"]
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    forest[i] = number.fit_transform(forest[i])
    forest[i] = number.fit_transform(forest[i])

#Checking Dataset 
forest.head()

forest.month.value_counts()

forest.day.value_counts()

forest.size_category.value_counts().plot(kind="bar")

#Splitting dataset as train and test
train,test = train_test_split(forest,test_size = 0.3,random_state=30)
trainX = train.drop(["size_category"],axis=1)
trainY = train["size_category"]
testX = test.drop(["size_category"],axis=1)
testY = test["size_category"]

#defining function prep_model for model builiding in neural network
def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1],kernel_initializer="normal",activation="sigmoid"))
    model.compile(loss="binary_crossentropy",optimizer = "rmsprop",metrics = ["accuracy"])
    return model

#Model building
first_model = prep_model([30,50,40,20,1]) #no of predictors=30, three hidden layers 50,40,20, no of target 1
trainX = np.asarray(trainX).astype('float32')
trainY = np.asarray(trainY).astype('float32')
first_model.fit(trainX,trainY,epochs=200)
pred_train = first_model.predict(trainX)



#Data Preprocessing again for class variable
pred_train = pd.Series([i[0] for i in pred_train])

forest_class = ["small","large"]
pred_train_class = pd.Series(["small"]*361)

pred_train_class

pred_train

pred_train_class[[i>0.5 for i in pred_train]] = "large"

#Confusion Matrix and checking accuracy
from sklearn.metrics import confusion_matrix
train["original_class"] = "small"
train.loc[train.size_category==1,"original_class"] = "large"
train.original_class.value_counts()
confusion_matrix(pred_train_class,train.original_class)

np.mean(pred_train_class==pd.Series(train.original_class).reset_index(drop=True))
#accuracy=0.99

pd.crosstab(pred_train_class,pd.Series(train.original_class).reset_index(drop=True))

#Testing Model on Test dataset and checking its accuracy
pred_test = first_model.predict(np.array(testX).astype("float32"))

pred_test = pd.Series([i[0] for i in pred_test])
pred_test

pred_test_class = pd.Series(["small"]*156)
pred_test_class[[i>0.5 for i in pred_test]] = "large"
test["original_class"] = "small"
test.loc[test.size_category==1,"original_class"] = "large"
test.original_class.value_counts()

np.mean(pred_test_class==pd.Series(test.original_class).reset_index(drop=True)) # 96.15

pd.crosstab(pred_test_class,test.original_class)
pd.crosstab(test.original_class,pred_test_class).plot(kind="bar")

#model visualiztion
from keras.utils import plot_model
plot_model(first_model,to_file="first_model.png")