from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from esnn import ESNN
from encoder import Encoder

X, y = load_iris(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

encoder = Encoder(20, 1.5, 0, 8)
esnn = ESNN(encoder,
            m=0.9, c=0.7, s=0.6)
esnn.train(X_train, y_train)
y_pred = esnn.test(X_test)

acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc}")
