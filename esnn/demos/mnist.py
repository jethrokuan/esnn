from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from esnn.esnn import ESNN
from esnn.encoder import Encoder

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

encoder = Encoder(10, 0.9, 0, 255)
esnn = ESNN(encoder, m=0.9, c=0.6, s=0.6)
esnn.train(X_train, y_train)
y_pred = esnn.test(X_test)

acc = accuracy_score(y_test, y_pred)

print(f"Neuron Count: {len(esnn.all_neurons)}")
print(f"Accuracy: {acc}")
