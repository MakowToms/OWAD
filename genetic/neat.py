from genetic.neuron import NeuronContainer
from genetic.connection import ConnectionContainer
from genetic.network import Network
import numpy as np
import copy


class NEAT:
    def __init__(self, n_input, n_output, n, score_function, data):
        self.score_function = score_function
        self.neurons = NeuronContainer(n_input, n_output)
        self.connections = ConnectionContainer(self.neurons.inputs, self.neurons.outputs)
        self.neurons.set_connection_container(self.connections)
        self.n = n
        self.population = self.create_population()
        self.data = data

    def create_population(self):
        population = []
        for i in range(self.n):
            population.append(Network(self.neurons.base_neurons, self.connections.base_connections, self.neurons, self.connections, self.score_function))
        return population

    def get_best_population(self):
        results = np.zeros([self.n])
        for index, pop in enumerate(self.population):
            results[index] = pop.score(self.data)
        print(f'Best accuracy: {results.max()}')
        best_pop = results.argsort()[::-1]
        return best_pop[:int(self.n/2)]

    def epoch(self):
        new_population_index = self.get_best_population()
        copied_population = copy.deepcopy(self.population)
        new_population = []
        for i in new_population_index:
            new_population.append(self.population[i])
            if np.random.random(1) > 0.2:
                new_population[-1].mutate_weight()
        for i in new_population_index:
            new_population.append(copied_population[i])
            if np.random.random(1) > 0.2:
                new_population[-1].mutate_weight()
        self.population = new_population


# trying on easy data

from neural_network.data_manipulation import load_easy_data
train, test = load_easy_data(True)


# create score function for easy
def score_function(res):
    return (np.sum(np.logical_and(res[2] > 0.5, train[:, 2] == 1)) +\
    np.sum(np.logical_and(res[2] <= 0.5, train[:, 2] == 0))) / res[2].shape[0]


data = {0: train[:, 0], 1: train[:, 1]}

# initialize neat wth 10 population size
neat = NEAT(2, 1, 10, score_function, data)
# results = neat.get_best_population()

# train
for i in range(100):
    neat.epoch()
# after around 20 epochs there is around 0.97 accuracy
