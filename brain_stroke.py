import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


df = pd.read_csv("brain_stroke.csv")

# def read_data(path):
#     return pd.read_csv(path)

print ("Rows     : " , df.shape[0])
print ("Columns  : " , df.shape[1])
print ("\nFeatures : \n" , df.columns.tolist())
# print ("\nMissing values :  ", df.isnull().sum().values.sum())
# print ("\nUnique values :  \n",df.nunique())

df["Residence_type"] = df["Residence_type"].apply(lambda x: 1 if x=="Urban" else 0)
df["ever_married"] = df["ever_married"].apply(lambda x: 1 if x=="Yes" else 0)
df["gender"] = df["gender"].apply(lambda x: 1 if x=="Male" else 0)
 
df = pd.get_dummies(data=df, columns=['smoking_status'])
df = pd.get_dummies(data=df, columns=['work_type'])

std=StandardScaler()
columns = ['avg_glucose_level','bmi','age']
scaled = std.fit_transform(df[['avg_glucose_level','bmi','age']])
scaled = pd.DataFrame(scaled,columns=columns)
df=df.drop(columns=columns,axis=1)

df=df.merge(scaled, left_index=True, right_index=True, how = "left")

X = df.drop(['stroke'], axis=1).values 
y = df['stroke'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def train_model(model):
    print("*"*70)
    print("Model:\n", model)
    model.fit(X_train, y_train)
    
    # training and testing accuracy
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
 
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    print(f"Training Score: {train_score:.2f}")
    print(f"Test Score: {test_score:.2f}")
    
    print("Confusion Matrix\n", cm)
    TN=cm[0,0]
    TP=cm[1,1]
    FN=cm[1,0]
    FP=cm[0,1]
    sensitivity=TP/float(TP+FN)
    specificity=TN/float(TN+FP)

    print('The accuracy of the model = TP+TN/(TP+TN+FP+FN) =       ',(TP+TN)/float(TP+TN+FP+FN),'\n',

    'The Missclassification = 1-Accuracy =                  ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',

    'Sensitivity or True Positive Rate = TP/(TP+FN) =       ',TP/float(TP+FN),'\n',

    'Specificity or True Negative Rate = TN/(TN+FP) =       ',TN/float(TN+FP),'\n')



svc = SVC(random_state=0)
train_model(svc)

forest = RandomForestClassifier(n_estimators = 100)
train_model(forest)

logreg = LogisticRegression()
train_model(logreg)

