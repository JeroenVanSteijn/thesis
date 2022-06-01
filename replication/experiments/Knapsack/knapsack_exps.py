import sys

sys.path.insert(0, "../..")
sys.path.insert(0, "../../QPTL/")
from SPO_dp_lr import *
import logging
from torch import optim

formatter = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(filename="weighted_QPTL.log", level=logging.INFO, format=formatter)

data = np.load("../../knapsackData.npz")
X_1gtrain = data["X_1gtrain"]
X_1gtest = data["X_1gtest"]
y_train = data["y_train"]
y_test = data["y_test"]
X_1gvalidation = X_1gtest[0:2880, :]
y_validation = y_test[0:2880]
y_test = y_test[2880:]
X_1gtest = X_1gtest[2880:, :]
weights = [data["weights"].tolist()]
weights = np.array(weights)

clf = SGD_SPO_dp_lr(
    weights=weights,
    epochs=10,
    optimizer=optim.Adam,
    capacity=[60],
    store_result=True,
    verbose=True,
    plotting=True,
)
pdf = clf.fit(X_1gtrain, y_train, X_1gvalidation, y_validation, X_1gtest, y_test)
print(pdf.head())
