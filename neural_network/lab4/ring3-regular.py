from neural_network.Network import Network
from neural_network.activations import *
from neural_network.loading_data import load_data, plot, one_hot_encode
from neural_network.testing_model import accuracy

# load dataset
train, test = load_data(file_name='rings3-balance', folder='classification', classification=True)


x = train[:, 0:2]
y = train[:, 2:3]
y_numeric = y
# plot(x[:, 0], x[:, 1], y[:, 0])
y = one_hot_encode(y)

# learn network without any momentum technique
network = Network(2, [50, 50], 3, [sigmoid, sigmoid, softmax], initialize_weights='Xavier')
mse = []
res = network.forward(x)
res = np.argmax(res, axis=1)
mse.append(accuracy(res, y_numeric))
for i in range(100):
    network.backward(x, y, eta=0.01, epochs=1, beta=0.01, moment_type='RMSProp')
    res = network.forward(x)
    res = np.argmax(res, axis=1)
    print('Iteracja {0}, Accuracy: {1:0.8f}'.format(i, accuracy(res, y_numeric)))
    mse.append(accuracy(res, y_numeric))

print('accuracy: ', accuracy(res, y_numeric))
plot(x[:, 0], x[:, 1], res)

