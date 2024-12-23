"""
The base algorithm for this project is taken from: 
https://github.com/sergi-s/Credit-Card-fraud-detection
"""

# libraries needed for the algorithm
import pandas as pd
import numpy as np
import tensorflow
import keras
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import itertools
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, RocCurveDisplay
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
import seaborn as sn
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE


"""
Random Forest
"""
def random_forest(X_train, X_test, y_train, y_test, X, y):
    
    random_forest = RandomForestClassifier(n_estimators=100)

    # Pandas Series.ravel() function returns the flattened underlying data as an ndarray.
    random_forest.fit(X_train,y_train.values.ravel())    # np.ravel() Return a contiguous flattened array

    y_pred = random_forest.predict(X_test)

    random_forest.score(X_test,y_test)

    return accuracy_score(y_test, y_pred)

"""
Decision trees
"""
def decision_trees(X_train, X_test, y_train, y_test, X, y):
    decision_tree = DecisionTreeClassifier()

    decision_tree.fit(X_train,y_train.values.ravel())

    y_pred = decision_tree.predict(X_test)

    decision_tree.score(X_test,y_test)
    
    return accuracy_score(y_test, y_pred)

"""
Neural network models
"""
def neural_networs(X_train, X_test, y_train, y_test, X, y):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model = Sequential([
        Dense(units=16, input_dim = 29,activation='relu'),   # input of 29 columns as shown above
        Dense(units=24,activation='relu'),
        Dropout(0.5),
        Dense(24,activation='relu'),
        Dense(24,activation='relu'),
        Dense(1,activation='sigmoid'),                        # binary classification fraudulent or not
    ])

    model.summary()

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(X_train,y_train,batch_size=15,epochs=5)

    score = model.evaluate(X_test, y_test)

    ## Confusion Matrix on unsee test set
    y_pred = model.predict(X_test)
    for i in range(len(y_test)):
        if y_pred[i]>0.5:
            y_pred[i]=1 
        else:
            y_pred[i]=0
  
    
    return accuracy_score(y_test, y_pred.round())

# """
# Weighted loss to account for large class imbalance in train dataset
# Called in neural_networs fuction
# """
# def weighted_loss(X_train, X_test, y_train, y_test, model, X, y, results_testset, results_fullset):
#     # class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), np.array([y_train[i][0] for i in range(len(y_train))]))
#     # class_weights = dict(enumerate(class_weights))
#     # class_weights
#     flat_y_train = np.ravel(y_train)  # Flatten y_train to ensure it's a 1D array if it's 2D

#     # Compute class weights
#     class_weights_array = class_weight.compute_class_weight(
#         class_weight='balanced', 
#         classes=np.unique(y_train), 
#         y=flat_y_train
#     )
#     # Convert to dictionary format
#     class_weights_dict = dict(zip(np.unique(y_train), class_weights_array))

#     model.fit(X_train, y_train, batch_size=15, epochs=5, class_weight=class_weights_dict, shuffle=True)

#     score_weighted = model.evaluate(X_test, y_test)

#     print(score_weighted)

#     ## Confusion Matrix on unseen test set
#     y_pred = model.predict(X_test)
#     for i in range(len(y_test)):
#         if y_pred[i]>0.5:
#             y_pred[i]=1 
#         else:
#             y_pred[i]=0
#     cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction
#     df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
#     # plt.figure(figsize = (10,7))
#     #sn.set(font_scale=1.4)
#     # sn.heatmap(df_cm, annot=True, fmt='g')
#     print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))

#     acc = accuracy_score(y_test, y_pred.round())
#     prec = precision_score(y_test, y_pred.round())
#     rec = recall_score(y_test, y_pred.round())
#     f1 = f1_score(y_test, y_pred.round())

#     # ### Store results in dataframe for comparing various Models
#     model_results = pd.DataFrame([['WeightedNeuralNetwork', acc, 1-rec, rec, prec, f1]],
#                columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])

#     # Append new results to the existing DataFrame using concat
#     results_testset = pd.concat([results_testset, model_results], ignore_index=True)

#     # Confusion matrix on the whole dataset
#     y_pred = model.predict(X)
#     y_expected = pd.DataFrame(y)
#     cnf_matrix = confusion_matrix(y_expected, y_pred.round())
#     # plot_confusion_matrix(cnf_matrix,classes=[0,1])
#     # plt.show()

#     acc = accuracy_score(y, y_pred.round())
#     prec = precision_score(y, y_pred.round())
#     rec = recall_score(y, y_pred.round())
#     f1 = f1_score(y, y_pred.round())

#     model_results = pd.DataFrame([['WeightedNeuralNetwork', acc, 1-rec, rec, prec, f1]],
#                 columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])

#     # Append new results to the existing DataFrame using concat
#     results_fullset = pd.concat([results_fullset, model_results], ignore_index=True)

#     return results_testset, results_fullset

# """
# Undersampling
# """
# def undersampling(df, X, y, results_testset, results_fullset):
#     fraud_indices = np.array(df[df.Class == 1].index)
#     number_records_fraud = len(fraud_indices)
#     print(number_records_fraud)

#     normal_indices = df[df.Class == 0].index

#     len(normal_indices)

#     # Random select N indices from non fraudulent samples (N equals to number of fraudulent records)
#     random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
#     random_normal_indices = np.array(random_normal_indices)
#     print(len(random_normal_indices))

#     under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
#     print(len(under_sample_indices))

#     under_sample_data = df.iloc[under_sample_indices,:]

#     X_undersample = under_sample_data.iloc[:,under_sample_data.columns != 'Class']
#     y_undersample = under_sample_data.iloc[:,under_sample_data.columns == 'Class']

#     X_train, X_test, y_train, y_test = train_test_split(X_undersample,y_undersample, test_size=0.3)

#     X_train = np.array(X_train)
#     X_test = np.array(X_test)
#     y_train = np.array(y_train)
#     y_test = np.array(y_test)

#     model = Sequential([
#         Dense(units=16, input_dim = 29,activation='relu'),   # input of 29 columns as shown above
#         Dense(units=24,activation='relu'),
#         Dropout(0.5),
#         Dense(24,activation='relu'),
#         Dense(24,activation='relu'),
#         Dense(1,activation='sigmoid'),                        # binary classification fraudulent or not
#     ])

#     model.summary()

#     model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#     model.fit(X_train,y_train,batch_size=15,epochs=5)

#     y_pred = model.predict(X_test)
#     y_expected = pd.DataFrame(y_test)
#     cnf_matrix = confusion_matrix(y_expected, y_pred.round())
#     # plot_confusion_matrix(cnf_matrix, classes=[0,1])
#     # plt.show()

#     acc = accuracy_score(y_test, y_pred.round())
#     prec = precision_score(y_test, y_pred.round())
#     rec = recall_score(y_test, y_pred.round())
#     f1 = f1_score(y_test, y_pred.round())

#     ### Store results in dataframe for comparing various Models
#     model_results = pd.DataFrame([['UnderSampledNeuralNetwork', acc, 1-rec, rec, prec, f1]],
#                 columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])
#     # Use pd.concat to combine the DataFrames
#     results_testset = pd.concat([results_testset, model_results], ignore_index=True)

#     # Confusion matrix on the whole dataset
#     y_pred = model.predict(X)
#     y_expected = pd.DataFrame(y)
#     cnf_matrix = confusion_matrix(y_expected, y_pred.round())
#     # plot_confusion_matrix(cnf_matrix, classes=[0,1])
#     # plt.show()

#     acc = accuracy_score(y, y_pred.round())
#     prec = precision_score(y, y_pred.round())
#     rec = recall_score(y, y_pred.round())
#     f1 = f1_score(y, y_pred.round())

#     model_results = pd.DataFrame([['UnderSampledNeuralNetwork', acc, 1-rec, rec, prec, f1]],
#                columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])

#     results_fullset = pd.concat([results_testset, model_results], ignore_index=True)

#     return results_testset, results_fullset

# """
# Oversampling with SMOTE
# """
# def oversampling_SMOTE(df, results_testset, results_fullset):
#     smote = SMOTE()
#     X = df.iloc[:, df.columns != 'Class']
#     y = df.iloc[:, df.columns == 'Class']  # Response variable determining if fraudulent or not
#     X_resample, y_resample = smote.fit_resample(X, y) 

#     print('Number of total transactions before SMOTE upsampling: ', len(y), '...after SMOTE upsampling: ', len(y_resample))
#     print('Number of fraudulent transactions before SMOTE upsampling: ', len(y[y.Class==1]), 
#         '...after SMOTE upsampling: ', np.sum(y_resample[y_resample==1]))


#     y_resample = pd.DataFrame(y_resample)
#     X_resample = pd.DataFrame(X_resample)

#     X_train, X_test, y_train, y_test = train_test_split(X_resample,y_resample,test_size=0.3)

#     X_train = np.array(X_train)
#     X_test = np.array(X_test)
#     y_train = np.array(y_train)
#     y_test = np.array(y_test)

#     model = Sequential([
#         Dense(units=16, input_dim = 29,activation='relu'),   # input of 29 columns as shown above
#         Dense(units=24,activation='relu'),
#         Dropout(0.5),
#         Dense(24,activation='relu'),
#         Dense(24,activation='relu'),
#         Dense(1,activation='sigmoid'),                        # binary classification fraudulent or not
#     ])

#     model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#     model.fit(X_train,y_train,batch_size=15,epochs=5) 

#     y_pred = model.predict(X_test)
#     y_expected = pd.DataFrame(y_test)
#     cnf_matrix = confusion_matrix(y_expected, y_pred.round())
#     # plot_confusion_matrix(cnf_matrix, classes=[0,1])
#     # plt.show()

#     acc = accuracy_score(y_test, y_pred.round())
#     prec = precision_score(y_test, y_pred.round())
#     rec = recall_score(y_test, y_pred.round())
#     f1 = f1_score(y_test, y_pred.round())

#     ### Store results in dataframe for comparing various Models
#     model_results = pd.DataFrame([['OverSampledNeuralNetwork', acc, 1-rec, rec, prec, f1]],
#                 columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])
#     results_testset = pd.concat([results_testset,model_results],ignore_index=True)

#     # Confusion matrix on the whole dataset
#     y_pred = model.predict(X)
#     y_expected = pd.DataFrame(y)
#     cnf_matrix = confusion_matrix(y_expected, y_pred.round())
#     # plot_confusion_matrix(cnf_matrix, classes=[0,1])
#     # plt.show()

#     acc = accuracy_score(y, y_pred.round())
#     prec = precision_score(y, y_pred.round())
#     rec = recall_score(y, y_pred.round())
#     f1 = f1_score(y, y_pred.round())

#     model_results = pd.DataFrame([['OverSampledNeuralNetwork', acc, 1-rec, rec, prec, f1]],
#                 columns = ['Model', 'Accuracy', 'FalseNegRate', 'Recall', 'Precision', 'F1 Score'])
#     results_fullset = pd.concat([results_fullset,model_results], ignore_index = True)

#     print(results_testset)



def main():
    """
    Loading and splitting data
    """
    print("loading csv...\n")
    # load the data into a dataframe
    df = pd.read_csv('creditcard.csv')

    # remove the Time column from the dataset
    df = df.drop(['Time'],axis=1)

    # split the dataset into fraud and not-fraud
    X = df.iloc[:, df.columns != 'Class']
    y = df.iloc[:, df.columns == 'Class']  # Response variable determining if fraudulent or not
    
    print("data loaded and split...\n")
    # split the fraud and not-fraud data into test and train data
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)

    print("starting random forest...\n")
    rf = random_forest(X_train, X_test, y_train, y_test, X, y)

    print("Starting decision trees...\n")
    dt = decision_trees(X_train, X_test, y_train, y_test, X, y)

    print("starting neural_networks...\n")
    nn = neural_networs(X_train, X_test, y_train, y_test, X, y)

    # print("Strting undersampling balancing...\n")
    # results_testset, results_fullset = undersampling(df, X, y, results_testset, results_fullset)

    # print("Strting undersampling balancing with SMOTE...\n")
    # oversampling_SMOTE(df, results_testset, results_fullset)

if __name__ == '__main__':
    main()
