import joblib


model = joblib.load('mnist_openheart.joblib')
W1 = model['W1']
W2 = model['W2']
W3 = model['W3']
b1 = model['b1']
b2 = model['b2']
b3 = model['b3']
print(W1)