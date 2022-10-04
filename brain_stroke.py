import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
import joblib

def read_data(path="brain_stroke.csv"):
    df = pd.read_csv(path)
    return df


def print_info(df):
    """
    This function prints a basic overview of a given pandas dataframe
    """
    print ("Rows     : " , df.shape[0])
    print ("Columns  : " , df.shape[1])
    print ("\nFeatures : \n" , df.columns.tolist())
    print ("\nMissing values :  ", df.isnull().sum().values.sum())
    print ("\nUnique values :  \n",df.nunique())


def preprocessing(df, ):
    preprocessing_dict = {}
    cols_to_label_encode = ['Residence_type', "ever_married", "gender", 'smoking_status', 'work_type']
    for col in cols_to_label_encode:
        lb = LabelEncoder()
        df[col] = lb.fit_transform(df[col])
        preprocessing_dict[col] = lb
    
    cols_to_stdscale = [] #['avg_glucose_level','bmi','age']
    for col in cols_to_stdscale:
        std=StandardScaler()
        df[col] = std.fit_transform(df[col])
        preprocessing_dict[col] = std
    
    joblib.dump(preprocessing_dict, 'preprocessing_columns.pkl')
    return df

def dataset_split(df):
    X = df.drop(['stroke'], axis=1).values
    y = df['stroke'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test


def confusion_matrix_info(cm):
    """
    This function will take as input argument a confusion matrix,
    and print the information like precision, recall, etc.
    """
    print("Confusion Matrix\n", cm)
    TN=cm[0,0]
    TP=cm[1,1]
    FN=cm[1,0]
    FP=cm[0,1]
    sensitivity=TP/float(TP+FN)
    accuracy = (TP+TN)/(TP+TN+FN+FP)
    precision = TP/float(TP+FP)
    recall = TP/float(TP+FN)
    specificity=TN/float(TN+FP)

    print(f'The accuracy of the model = TP+TN/(TP+TN+FP+FN) = {accuracy:.2f}\n',
          f'The Missclassification = 1-Accuracy             = {1-accuracy:.2f}\n',
          f'The Precision                                   = {precision:.2f}\n',
          f'The recall                                      = {recall:.2f}\n',
          f'Sensitivity or True Positive Rate = TP/(TP+FN)  = {sensitivity:.2f}\n',
          f'Specificity or True Negative Rate = TN/(TN+FP)  = {specificity:.2f}\n')


def train_model(model, modelname, X_train, y_train, X_test, y_test):
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
    
    confusion_matrix_info(cm)
    joblib.dump(model, modelname + ".pkl")


def hyperparam_tuning(model, params_grid, modelname, X_train, y_train, X_test, y_test):
    model_rndm_tuned = RandomizedSearchCV(estimator = model,
                               param_distributions = params_grid,
                               n_iter = 100, 
                               cv = 5,
                               verbose=2, 
                               random_state=35, 
                               n_jobs = -1)
    model_rndm_tuned.fit(X_train, y_train)
    # training and testing accuracy
    train_score = model_rndm_tuned.score(X_train, y_train)
    test_score = model_rndm_tuned.score(X_test, y_test)
 
    y_pred = model_rndm_tuned.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    print(f"Training Score: {train_score:.2f}")
    print(f"Test Score: {test_score:.2f}")
    
    confusion_matrix_info(cm)
    joblib.dump(model_rndm_tuned, modelname + ".pkl")


def predict_new_data(data_path, model_path):
    df = read_data(data_path)
    df = preprocessing(df)
    model = joblib.load(model_path)
    predictions = model.predict(df)


if __name__ == "__main__":
    df = read_data("brain_stroke.csv")
    df = preprocessing(df)
    X_train, X_test, y_train, y_test = dataset_split(df)

    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train.ravel())

    svc = SVC(random_state=0)
    train_model(svc, "svc", X_train, y_train, X_test, y_test)

    forest = RandomForestClassifier(max_depth=3, n_estimators = 10)
    train_model(forest, "forest", X_train, y_train, X_test, y_test)

    logreg = LogisticRegression()
    train_model(logreg, "logreg", X_train, y_train, X_test, y_test)

    # hyperparam tuning #

    n_estimators =[64,100,128,200]
    max_features = [2,3,5,7]
    bootstrap = [True,False]

    param_grid = {'n_estimators':n_estimators,
                'max_features':max_features,
                'bootstrap':bootstrap}

    forest = RandomForestClassifier(max_depth=3, n_estimators = 10)
    hyperparam_tuning(forest, param_grid, "forest_tuned", 
                        X_train, y_train, X_test, y_test)