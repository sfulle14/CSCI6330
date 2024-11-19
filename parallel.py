"""
The base algorithm for this project is taken from:
https://github.com/sergi-s/Credit-Card-fraud-detection
"""

# libraries needed for the algorithm
import polars
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, RocCurveDisplay

# used to run program in parallel
import concurrent.futures
from neural_network import train_neural_network

"""
Random Forest
Can be made more parallel by adding the n_jobs varibale to the RandomForestClassifier.
"""
def random_forest(X_train, X_test, y_train, y_test):

    # n_jobs is the parallel input variable and -1 means use max processes.
    random_forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    random_forest.fit(X_train, y_train)

    y_pred = random_forest.predict(X_test)

    random_forest.score(X_test, y_test)

    return accuracy_score(y_test, y_pred)


"""
Decision trees
Cannot be made more parallel since it is just one tree.
"""
def decision_trees(X_train, X_test, y_train, y_test):
    decision_tree = DecisionTreeClassifier()

    decision_tree.fit(X_train, y_train)

    y_pred = decision_tree.predict(X_test)

    decision_tree.score(X_test,y_test)

    return accuracy_score(y_test, y_pred)


def main():
    """
    Loading and splitting data
    """

    print("loading csv...\n")

    df = polars.read_csv("creditcard.csv", schema_overrides={
                         "Time": polars.Utf8}).drop("Time")

    X = df.select(polars.all().exclude("Class")).to_numpy()
    y = df.select("Class").to_numpy().flatten()

    print("data loaded and split...\n")

	# split the features and labels into training and testing portions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    print("starting model training in parallel...\n")

    # use concurrent.futures to run the processes in parallel
    # https://docs.python.org/3/library/concurrent.futures.html
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit tasks to the executor
        future_rf = executor.submit(random_forest, X_train, X_test, y_train, y_test)
        future_dt = executor.submit(decision_trees, X_train, X_test, y_train, y_test)
        future_nn = executor.submit(train_neural_network, X_train, X_test, y_train,  y_test)

        print("waiting for results.\n")
        rf_acc = future_rf.result()
        dt_acc = future_dt.result()
        nn_acc = future_nn.result()

if __name__ == '__main__':
    main()
