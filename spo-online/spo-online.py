import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

training_set = pd.read_csv("../data/uncertain-processing-time/train.csv")
validation_set = pd.read_csv("../data/uncertain-processing-time/validate.csv")

x_train = training_set.iloc[:,0:-1].values
y_train = training_set.iloc[:,-1].values
x_val = validation_set.iloc[:,0:-1].values
y_val = validation_set.iloc[:,-1].values

classifier = MLPClassifier(hidden_layer_sizes=(150, 100, 50),
                           max_iter=300,
                           activation='relu',
                           solver='adam',
                           random_state=1)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_val)

# Comparing the predictions against the actual observations in y_val
cm = confusion_matrix(y_pred, y_val)

print("Accuracy of MLPClassifier: ", accuracy(cm))

