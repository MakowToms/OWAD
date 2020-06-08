from neural_network.data_manipulation import load_easy_data
import numpy as np
from genetic.neat import NEAT
import copy
train, test = load_easy_data(True)


# create score function for easy
def score_function(res):
    return (np.sum(np.logical_and(res[2] > 0.5, train[:, 2] == 1)) +\
    np.sum(np.logical_and(res[2] <= 0.5, train[:, 2] == 0))) / res[2].shape[0]


data = {0: train[:, 0], 1: train[:, 1]}

# initialize neat wth 100 population size
neat = NEAT(2, 1, 100, score_function, data)
# results = neat.get_best_population()

# train
for i in range(20):
    neat.epoch()


# neat.get_best_population()
# print(neat.population[0].score(copy.deepcopy(data)))
# neat.population[0].connection_ids
# neat.population[0].neuron_ids
# [(k, v.active, v.weight) for k, v in neat.population[1].connection_data.items()]
# neat.population[15].connection_container.possible_connections[4]
# neat.connections.base_connections
# neat.neurons.connection_container
# after around 20 epochs there is around 0.97 accuracy
