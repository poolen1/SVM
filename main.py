
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from prettytable import PrettyTable


def get_predictions(train_set, train_labels, test, the_kernel):
    y_predictions = []

    print("test: ", test.shape)
    for i in range(len(train_set)):
        print("train: ", train_set[i].shape)

    for i in range(len(train_set)):
        X = train_set[i]
        y = train_labels[i]
        test = train_set[i-1]

        svc = SVC(kernel=the_kernel)
        svc.fit(X, y)
        y_predictions.append(svc.predict(test))

    return y_predictions


def calc_metrics(y_predictions, test_set, test_labels):
    # Calculate evaluation metrics
    accuracies = []
    for i in range(len(y_predictions)):
        X_test = test_set
        y_groundtruth = test_labels.astype(int)

        y_pred = y_predictions[i]

        # Calc metrics
        acc = sum(y_pred == y_groundtruth) / len(y_groundtruth)
        acc = acc * 100
        acc = round(acc, 2)
        accuracies.append(acc)

    return accuracies


def calc_avg_acc(acc_set):
    total = sum(acc_set)
    avg = total/len(acc_set)
    avg = round(avg, 2)
    acc_set.append(avg)


input_file = "MNIST_HW4.csv"

data_set = pd.read_csv(input_file)

# print(data_set)

# n == # of samples, ie, rows
# p == # of features, ie, cols
n, p = data_set.shape

# less 1 to account for labels column
p -= 1

# Encode labels as 0 and 1
y = data_set.iloc[:, 0]

# Separate labels
X = data_set.iloc[:, 1:-1]
X = pd.DataFrame(np.c_[np.ones(n), X])

# Normalize data
X = X/255

# Split into 5 folds for cross-validation
folds = np.array_split(X, 6)

# Split labels for cross-validation
label_folds = np.array_split(y, 6)

# Training Data
training_sets = folds[:-1]
training_labels = label_folds[:-1]

# Test Data
test_set = folds[-1]
test_labels = label_folds[-1]

print(test_set.shape)

predictions = []
kernels = ['linear', 'poly', 'rbf']
for i in range(len(kernels)):
    predictions.append(get_predictions(training_sets, training_labels, test_set, kernels[i]))

accuracy_set = []
for i in range(len(predictions)):
    accuracies = calc_metrics(predictions[i], test_set, test_labels)
    accuracy_set.append(accuracies)

for i in range(len(accuracy_set)):
    calc_avg_acc(accuracy_set[i])

table = PrettyTable(['Test#', 'Linear', 'Poly', 'RBF'])
length = len(accuracy_set[0]) - 1
for i in range(length):
    table.add_row([i+1, accuracy_set[0][i], accuracy_set[1][i], accuracy_set[2][i]])

table.add_row(["Acc", accuracy_set[0][length], accuracy_set[1][length], accuracy_set[2][length]])

print(table)
