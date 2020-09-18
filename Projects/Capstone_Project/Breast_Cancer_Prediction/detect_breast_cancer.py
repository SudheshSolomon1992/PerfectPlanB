import pickle
import numpy as np
import pandas as pd
import sklearn.datasets
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
# Feature Selection
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
# Models
import xgboost as xgb
from xgboost import plot_tree
from xgboost import plot_importance
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline 
# Evaluation
from sklearn.metrics import accuracy_score

global selected_columns

# Feature Selection Techniques
def filter_selection(data):
    corrmat = data.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    #plot heat map
    g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
    # plt.show()

    #Correlation with output variable
    cor_target = abs(corrmat["class"])
    #Selecting highly correlated features
    relevant_features = cor_target[cor_target>0.5]
    print (relevant_features)

def rfe_selection(model, data, X, y, X_train, X_test, y_train, y_test):
    #no of features
    nof_list=np.arange(1,X.shape[1])       
    high_score=0
    #Variable to store the optimum features
    nof=0           
    score_list =[]
    data_X = data.drop('class', 1)
    for n in range(len(nof_list)):
        # model = SGDClassifier()
        rfe = RFE(model,nof_list[n])
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)
        if(score>high_score):
            high_score = score
            nof = nof_list[n]
    print("Optimum number of features: %d" %nof)
    print("Score with %d features: %f" % (nof, high_score))
    selected_features = optimum_rfe(model, data_X, X, y, nof)
    return selected_features.tolist()

def optimum_rfe(model, data_X, X, y, nof):
    cols = list(data_X.columns)
    #Initializing RFE model
    rfe = RFE(model, nof)       
    #Transforming data using RFE
    X_rfe = rfe.fit_transform(X,y)   
    #Fitting the data to model
    model.fit(X_rfe,y)                
    temp = pd.Series(rfe.support_,index = cols)
    selected_features_rfe = temp[temp==True].index
    return selected_features_rfe

def embedded_selection(data, X, y):
    reg = LassoCV()
    reg.fit(X, y)
    data_X = data.drop('class', 1)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" %reg.score(X,y))
    coef = pd.Series(reg.coef_, index = data_X.columns)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    imp_coef = coef.sort_values()

    plt.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind = "barh")
    plt.title("Feature importance using Lasso Model")
    plt.show()

def create_optimal_dataset(columns):
    # Load the dataset
    breast_cancer = sklearn.datasets.load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target

    data = pd.DataFrame(X, columns=breast_cancer.feature_names)
    data = data.drop(data.columns.difference(columns), 1, inplace=True)
    data['class'] = breast_cancer.target
    return data, X, Y

def getDataset():
    # Load the dataset
    breast_cancer = sklearn.datasets.load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target

    # prints the number of rows and columns
    # print (X.shape)
    # print (y.shape)

    data = pd.DataFrame(X, columns=breast_cancer.feature_names)
    data['class'] = breast_cancer.target

    # prints the first 5 records in the dataframe.
    # print (data.head())
    # prints the statistical information about the dataset.
    # print (data.describe())
    # Groups data based on classes and gives the count in each class.
    # print (data['class'].value_counts())
    # gives the class labels.
    # print (breast_cancer.target_names)
    # Groups data based on classes and gives the mean of feature of each class.
    # print (data.groupby('class').mean())
    return data, X, y

def splitDataset(data, X, y):
    # Split the data into train and test.

    # X_train : Training data.
    # X_test : Testing data.
    # y_train : Training labels.
    # y_test : Testing labels.
    # test_size: specify the size of the test data.
    # stratify: use if y.shape, y_train.shape, y_test.shape is not nearly equal. -- Use only for y not needed for X.
    # random_state: previnting the value from changing on each run.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1)
    # print (y.shape, y_train.shape, y_test.shape)
    # print (y.mean(), y_train.mean(), y_test.mean())
    
    # Hyper-parameter Tuning
    classifier = grid_search(X_train, y_train)
    return classifier, X_train, X_test, y_train, y_test

# Hyperparameter tuning
def grid_search(features, target):
    pipe = Pipeline([("classifier", SGDClassifier())])

    # Create dictionary with candidate learning algorithms and their hyperparameters
    search_space = [
                    {"classifier": [SGDClassifier()],
                    "classifier__loss": ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                    "classifier__penalty":['l1', 'l2', 'elasticnet'],
                    "classifier__max_iter":[250, 500, 750, 1000],
                    "classifier__shuffle": [True, False],
                    "classifier__early_stopping": [True, False],
                    "classifier__n_iter_no_change": [5,10]
                    },
                    {"classifier": [LogisticRegression()],
                    "classifier__penalty": ['l2','l1'],
                    "classifier__C": np.logspace(0, 4, 10),
                    "classifier__fit_intercept":[True, False],
                    "classifier__solver":['saga','liblinear']
                    },
                    {"classifier": [LogisticRegression()],
                    "classifier__penalty": ['l2'],
                    "classifier__C": np.logspace(0, 4, 10),
                    "classifier__solver":['newton-cg','saga','sag','liblinear'], ##These solvers don't allow L1 penalty
                    "classifier__fit_intercept":[True, False]
                    },
                    {"classifier": [RandomForestClassifier()],
                    "classifier__n_estimators": [10, 100, 1000],
                    "classifier__max_depth":[5,8,15,25,30,None],
                    "classifier__min_samples_leaf":[1,2,5,10,15,100],
                    "classifier__max_leaf_nodes": [2, 5,10]
                    },
                    {"classifier": [xgb.XGBClassifier()],
                    "classifier__learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
                    "classifier__max_depth": [ 3, 4, 5, 6, 8, 10, 12, 15],
                    "classifier__min_child_weight": [ 1, 3, 5, 7 ],
                    "classifier__gamma": [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
                    "classifier__colsample_bytree": [ 0.3, 0.4, 0.5 , 0.7 ]}]
    # create a gridsearch of the pipeline, the fit the best model
    gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0,n_jobs=-1) # Fit grid search
    best_model = gridsearch.fit(features, target)

    print(best_model.best_estimator_)
    print("The mean accuracy of the model is:",best_model.score(features, target))
    
    # return the best model with the best set of parameters
    return best_model.best_params_['classifier']

def trainModel(filename, classifier, X_train, y_train):
    # training the model on the training data
    classifier.fit(X_train, y_train)
    # save the model to disk
    pickle.dump(classifier, open(filename, 'wb'))
    return classifier

def evaluate_model(pickle_filename, data, X_train, X_test, y_train, y_test):

    # load the model from disk
    classifier = pickle.load(open(pickle_filename, 'rb'))
    result = classifier.score(X_test, y_test)
    # print(result)

    # Model Evaluation
    
    # Prediction on training data
    prediction_train = classifier.predict(X_train)
    accuracy_train = accuracy_score(y_train, prediction_train)
    print ('Accuracy on training data', accuracy_train)
    # Prediction on test data
    prediction_test = classifier.predict(X_test)
    accuracy_test = accuracy_score(y_test, prediction_test)
    print ('Accuracy on test data', accuracy_test)

def predict_record(data, classifier):
    row_no = 1

    # print (data.columns)
    # print (data.head(2))

    # print (data['mean texture'].head(row_no), data['mean area'].head(row_no), data['worst smoothness'].head(row_no), data['mean compactness'].head(row_no), data['mean concave points'].head(row_no), data['area error'].head(row_no), data['worst radius'].head(row_no), data['worst texture'].head(row_no), data['worst perimeter'].head(row_no), data['worst area'].head(row_no), data['worst smoothness'].head(row_no), data['worst compactness'].head(row_no), data['worst concavity'].head(row_no), data['worst concave points'].head(row_no), data['class'].head(1))

    # Detecting whether the patient has breast cancer in malignant or benign stage
    input_data = (10.38,1001.0,0.1622,0.2776,0.1471,153.4,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654)
    # input_data_class = 0

    # change the input_data tuple into a numpy array
    input_array = np.asarray(input_data)

    # reshaping the array as we are predicting the data for one instance
    input_array_reshaped = input_array.reshape(1, -1)

    # predict the class (malignant or benign)
    prediction = classifier.predict(input_array_reshaped)
    
    # prints a list with element [0] if malignant or [1] if benign
    # print (prediction)

    # 0: malignant
    # 1: benign

    if (prediction[0] == 0):
        print ('The breast cancer is malignant')
    else:
        print ('The breast cancer is benign')

def main():

    pickle_filename = './breast_cancer_predictor_model.sav'

    # Get the dataset and split it to train/test data.
    data, X, y = getDataset()
    classifier, X_train, X_test, y_train, y_test = splitDataset(data, X, y)
    
    # Feature Selection - Use one of the three methods
    selected_columns = rfe_selection(classifier, data, X, y, X_train, X_test, y_train, y_test)
    # Get the feature names from the methods below manually and add them to selected columns list
    # filter_selection(data)
    # embedded_selection(data,X,y)

    # Create optimal dataset
    data_optimal = data[selected_columns]
    data_optimal['class'] = y
    # print (data_optimal.columns)

    X_optimal = data_optimal.iloc[:,:-1]
    y_optimal = data_optimal.iloc[:,-1]

    X_optimal = np.asarray(X_optimal)
    y_optimal = np.asarray(y_optimal)

    # Split the data
    X_train_optimal, X_test_optimal, y_train_optimal, y_test_optimal = train_test_split(X_optimal, y_optimal, test_size=0.1, stratify=y, random_state=1)

    # Build and train model
    classifier_optimal = trainModel(pickle_filename, classifier, X_train_optimal, y_train_optimal)

    # Evaluate the model
    evaluate_model(pickle_filename, data_optimal, X_train_optimal, X_test_optimal, y_train_optimal, y_test_optimal)

    # Make predictions
    predict_record(data_optimal, classifier)

if __name__ == "__main__":
    main()
