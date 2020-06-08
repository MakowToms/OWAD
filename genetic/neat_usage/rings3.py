from neural_network.data_manipulation import load_data
import numpy as np
from genetic.neat import NEAT
import copy
from neural_network.plots import plot_data_2d, plot_measure_results_data
import matplotlib.pyplot as plt

# load dataset
train, test = load_data(file_name='rings3-regular', folder='classification', classification=True)

# x and y - observations and values for them
x = train[:, 0:2]
y = train[:, 2:3]


# create score function for easy
def score_function(res):
    results = np.vstack([res[2], res[3], res[4]])
    return np.sum(train[:, 2] == np.argmax(results, axis=0)) / res[2].shape[0]


# res = {2: np.array([1,2,3]), 3: np.array([3,1,2]), 4: np.array([0,4,0])}
data = {0: train[:, 0], 1: train[:, 1]}

# initialize neat wth 100 population size
neat = NEAT(2, 3, 1000, score_function, data)
# results = neat.get_best_population()

# train
for i in range(50):
    neat.epoch()


best = neat.get_best_population()[0]
print(neat.population[best].score(copy.deepcopy(data)))
print(neat.population[best].connection_ids)
print(neat.population[best].neuron_ids)

# plot data classes
plt.figure(figsize=(6.4, 9.6))
plt.subplot(211)
plot_data_2d(x[:, 0], x[:, 1], y[:, 0], title='True classes of points on the plane', show=False)

res = neat.population[best].evaluate(copy.deepcopy(data))
results = np.vstack([res[2], res[3], res[4]])
y_predicted = np.argmax(results, axis=0)
plt.subplot(212)
plot_data_2d(x[:, 0], x[:, 1], y_predicted, title='Predicted classes of points on the plane', show=False)
plt.show()
# after around 50 epochs there is around 0.5 accuracy
# without species it is rather hard to train
