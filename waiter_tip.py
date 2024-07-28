# import dependencies.
import pandas as pd
import plotly.express as px
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR as svr
from sklearn.metrics import (
    r2_score,
    mean_absolute_error as mae,
    mean_squared_error as mse,
)
import pickle
# load the dataset
df = pd.read_csv('tips.csv')

# EDA(exploratory data analysis).
print(df.head())
print(df.tail())
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.describe())

scatter = sns.scatterplot(x="total_bill",y="tip",data = df,size="size")
plt.show()
# showing the four continous attribute through scatter plot and donut chart.
sex_scatter = px.scatter(data_frame = df, x = "total_bill", y = "tip", size = "size", color = "sex")
sex_pie = px.pie(df, values = "tip", names = "sex", hole = 0.6)
# sex_scatter.show()
# sex_pplie.show()

smoker_scatter = px.scatter(data_frame = df, x = "total_bill", y = "tip", size = "size", color = "smoker")
smoker_pie = px.pie(df, values = "tip", names = "smoker", hole = 0.6)
# smoker_scatter.show()
# smoker_pie.show()

day_scatter = px.scatter(data_frame = df, x = "total_bill", y = "tip", size = "size", color = "day")
day_pie = px.pie(df, values = "tip", names = "day", hole = 0.6)
# day_scatter.show()
# day_pie.show()

time_scatter = px.scatter(data_frame = df, x = "total_bill", y = "tip", size = "size", color = "time")
time_pie = px.pie(df, values = "tip", names = "time", hole = 0.6)
# time_scatter.show()
# time_pie.show()

# feature engineering
le = LabelEncoder()
# handling sex feature.
print(df.sex.unique())
df["sex"] = le.fit_transform(df["sex"])
# handling somking feature.
print(df.smoker.unique())
df["smoker"] = le.fit_transform(df["smoker"])
# handling day feature.
print(df.day.unique())
days = pd.get_dummies(df.day, drop_first = True)
df = pd.concat([df,days],axis=1)
df.drop(["day"],axis = 1, inplace = True)
# handling time feature.
print(df.time.unique())
df["time"] = le.fit_transform(df["time"])

# Correlation
cor_relation = df.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(cor_relation,annot = True,cmap="coolwarm")
plt.show()
# Creating dependent and Independent variables.
X = df.drop(['tip'], axis = 1)
Y = df["tip"]
xData = X.values
yData = Y.values
print(X.shape)
print(Y.shape)

# We will create a dictionary for each different models. 
# Each dictionary will have keys for model, its prediction value,
# r2-score, mean squre error, root mean square error, and mean absolute error.
lr_dict = {
    "model":[],
    "prediction":[],
    "r2_score":[],
    "mse":[],
    "rmse":[],
    "mae":[]
}
rfr_dict = {
    "model":[],
    "prediction":[],
    "r2_score":[],
    "mse":[],
    "rmse":[],
    "mae":[]
}
svm_dict = {
    "model":[],
    "prediction":[],
    "r2_score":[],
    "mse":[],
    "rmse":[],
    "mae":[]
}
knn_dict = {
    "model":[],
    "prediction":[],
    "r2_score":[],
    "mse":[],
    "rmse":[],
    "mae":[]
}
stack_dict = {
    "model":[],
    "prediction":[],
    "r2_score":[],
    "mse":[],
    "rmse":[],
    "mae":[]
}
# we have to store our testing set of each variation to compare with predictions of the models.
# so we will create a list to store this.
yTest_list = []

#Train and test data on different variation
# varition 1 : 80-20 split
# varition 2 : 75-25 split
# varition 3 : 70-30 split
for i in range(3):
    # spliting data into training and testing columns.
    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size = 0.20 + (i*0.05),random_state = 31)
    # storing yTest in the list (yTest_list).
    yTest_list.append(yTest)  

    # making use of Linear regression model. 
    lr = LinearRegression()                                      
    lr.fit(xTrain, yTrain)
    lr_dict["prediction"].append(lr.predict(xTest))
    lr_dict["model"].append(lr)
    # storing r2-score,mse,rmse and mae.
    lr_dict["r2_score"].append(r2_score(yTest,lr_dict["prediction"][i]))
    lr_dict["mse"].append(mse(yTest,lr_dict["prediction"][i]))
    lr_dict["mae"].append(mae(yTest,lr_dict["prediction"][i]))
    # since, rmse is simply the square root of mse. so, we will just find square root
    # of mse using math function.
    lr_dict["rmse"].append(math.sqrt(mse(yTest_list[i],lr_dict["prediction"][i])))
    
    # making use of Random Forest regressor model. 
    rfr = RandomForestRegressor()                    
    rfr.fit(xTrain, yTrain)
    rfr_dict["prediction"].append(rfr.predict(xTest))
    rfr_dict["model"].append(rfr)
    # storing r2-score,mse,rmse and mae.
    rfr_dict["r2_score"].append(r2_score(yTest,rfr_dict["prediction"][i]))
    rfr_dict["mse"].append(mse(yTest,rfr_dict["prediction"][i]))
    rfr_dict["mae"].append(mae(yTest,rfr_dict["prediction"][i]))
    # since, rmse is simply the square root of mse. so, we will just find square root
    # of mse using math function.
    rfr_dict["rmse"].append(math.sqrt(mse(yTest,rfr_dict["prediction"][i])))
       
    # making use of SVM model. 
    svm = svr(kernel="linear",C=1,gamma="auto")                    
    svm.fit(xTrain, yTrain)
    svm_dict["prediction"].append(svm.predict(xTest))
    svm_dict["model"].append(svm)
    # storing r2-score,mse,rmse and mae.
    svm_dict["r2_score"].append(r2_score(yTest,svm_dict["prediction"][i]))
    svm_dict["mse"].append(mse(yTest,svm_dict["prediction"][i]))
    svm_dict["mae"].append(mae(yTest,svm_dict["prediction"][i]))
    # since, rmse is simply the square root of mse. so, we will just find square root
    # of mse using math function.
    svm_dict["rmse"].append(math.sqrt(mse(yTest,svm_dict["prediction"][i])))
    
    # making use of K-Nearest Neighbour(KNN) model. 
    knn = KNeighborsRegressor(n_neighbors=3)                    
    knn.fit(xTrain, yTrain)
    knn_dict["prediction"].append(knn.predict(xTest))
    knn_dict["model"].append(knn)
    # storing r2-score,mse,rmse and mae.
    knn_dict["r2_score"].append(r2_score(yTest,knn_dict["prediction"][i]))
    knn_dict["mse"].append(mse(yTest,knn_dict["prediction"][i]))
    knn_dict["mae"].append(mae(yTest,knn_dict["prediction"][i]))
    # since, rmse is simply the square root of mse. so, we will just find square root
    # of mse using math function.
    knn_dict["rmse"].append(math.sqrt(mse(yTest,knn_dict["prediction"][i])))
    
#Creating Base model for stacking 
base_models = [
    ('rf', RandomForestRegressor()),
    ('gb', GradientBoostingRegressor())
]
# Creating meta model for stacking
meta_model = LinearRegression()
# Create the stacking regressor
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
for i in range(3):
    # spliting data into training and testing columns.
    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size = 0.20 + (i*0.05),random_state = 31)

    # making use of Stacking model. 
    stacking_model.fit(xTrain, yTrain)
    stack_dict["prediction"].append(stacking_model.predict(xTest))
    stack_dict["model"].append(stacking_model)
    # storing r2-score,mse,rmse and mae.
    stack_dict["r2_score"].append(r2_score(yTest,stack_dict["prediction"][i]))
    stack_dict["mse"].append(mse(yTest,stack_dict["prediction"][i]))
    stack_dict["mae"].append(mae(yTest,stack_dict["prediction"][i]))
    # since, rmse is simply the square root of mse. so, we will just find square root
    # of mse using math function.
    stack_dict["rmse"].append(math.sqrt(mse(yTest,stack_dict["prediction"][i])))
    
# Evaluation of the models.

# printing Evaluation report of Linear regression (r2-score, mean_square_error,
# root_mean_sqare_error and mean_absolute_error.)
for i in range(3):
    print(f"\nTrain Test split at {80-(i*5)}-{20+(i*5)}")
    print("r2 score of Linear regression model              :",lr_dict["r2_score"][i])
    print("mean square error of Linear regression model     :",lr_dict["mse"][i])
    print("Root mean square error of Linear regression model:",lr_dict["rmse"][i])
    print("mean absolute error of Linear regression model   :",lr_dict["mae"][i])   
# printing Evaluation report of Random Forest Regressor (r2-score, mean_square_error,
# root_mean_sqare_error and mean_absolute_error.)
for i in range(3):
    print(f"\nTrain Test split at {80-(i*5)}-{20+(i*5)}")
    print("r2 score of Random Forest model              :",rfr_dict["r2_score"][i])
    print("mean square error of Random Forest model     :",rfr_dict["mse"][i])
    print("Root mean square error of Random Forest model:",rfr_dict["rmse"][i])
    print("mean absolute error of Random Forest model   :",rfr_dict["mae"][i])   
# printing Evaluation report of SVM (r2-score, mean_square_error,
# root_mean_sqare_error and mean_absolute_error.)
for i in range(3):
    print(f"\nTrain Test split at {80-(i*5)}-{20+(i*5)}")
    print("r2 score of SVM model              :",svm_dict["r2_score"][i])
    print("mean square error of SVM model     :",svm_dict["mse"][i])
    print("Root mean square error of SVM model:",svm_dict["rmse"][i])
    print("mean absolute error of SVM model   :",svm_dict["mae"][i])
# printing Evaluation report of K-Nearest Neighbour(KNN) (r2-score, mean_square_error,
# root_mean_sqare_error and mean_absolute_error.)
for i in range(3):
    print(f"\nTrain Test split at {80-(i*5)}-{20+(i*5)}")
    print("r2 score of KNN model              :",knn_dict["r2_score"][i])
    print("mean square error of KNN model     :",knn_dict["mse"][i])
    print("Root mean square error of KNN model:",knn_dict["rmse"][i])
    print("mean absolute error of KNN model   :",knn_dict["mae"][i])  
# printing Evaluation report of Stacking model (r2-score, mean_square_error,
# root_mean_sqare_error and mean_absolute_error.)
for i in range(3):
    print(f"\nTrain Test split at {80-(i*5)}-{20+(i*5)}")
    print("r2 score of stack model              :",stack_dict["r2_score"][i])
    print("mean square error of stack model     :",stack_dict["mse"][i])
    print("Root mean square error of stack model:",stack_dict["rmse"][i])
    print("mean absolute error of stack model   :",stack_dict["mae"][i])  
#comparing all variatins of different models through bar plot
for i in range(3):
    list1=['linearRegression','RandomForest','SVM','KNN','stack']
    list2=[lr_dict["r2_score"][i],rfr_dict["r2_score"][i],
           svm_dict["r2_score"][i],knn_dict["r2_score"][i],stack_dict["r2_score"][i]]
    df_Accuracy=pd.DataFrame({"Method Used":list1,"Accuracy":list2})
    chart=sns.barplot(x='Method Used',y='Accuracy',data=df_Accuracy,width = 0.5)
    # title vary depending on variation.
    plt.title(f"Split {80-(i*5)}-{20+(i*5)}")
    plt.show()    
# Showing The variance of different through distplot.
# Linear Regression
for i in range(3):
    plt.title(f"Linear regression {80-(i*5)}-{20+(i*5)} Split")
    sns.distplot(yTest_list[i] - lr_dict["prediction"][i])
    plt.show()  
# Random Forest.
for i in range(3):
    plt.title(f"Random Rorest {80-(i*5)}-{20+(i*5)} Split")
    sns.distplot(yTest_list[i] - rfr_dict["prediction"][i])
    plt.show()
# SVM.
for i in range(3):
    plt.title(f"SVM {80-(i*5)}-{20+(i*5)} Split")
    sns.distplot(yTest_list[i] - svm_dict["prediction"][i])
    plt.show()  
# KNN
for i in range(3):
    plt.title(f"KNN {80-(i*5)}-{20+(i*5)} Split")
    sns.distplot(yTest_list[i] - knn_dict["prediction"][i])
    plt.show()
# Stacking.
for i in range(3):
    plt.title(f"Stacking model {80-(i*5)}-{20+(i*5)} Split")
    sns.distplot(yTest_list[i] - knn_dict["prediction"][i])
    plt.show()

# Dumping all the models in the pickle file
    
with open('lr_model.pkl', 'wb') as f:
    pickle.dump(lr_dict["model"], f)
with open('rfr_model.pkl', 'wb') as f:
    pickle.dump(rfr_dict["model"], f)
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_dict["model"], f)
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn_dict["model"], f)
with open('stack_model.pkl', 'wb') as f:
    pickle.dump(stack_dict["model"], f)