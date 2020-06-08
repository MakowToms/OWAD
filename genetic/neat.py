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
            results[index] = pop.score(copy.deepcopy(self.data))
        print(f'Best accuracy: {results.max():0.03}')
        best_pop = results.argsort()[::-1]
        return best_pop

    def epoch(self):
        new_population_index = self.get_best_population()
        copied_population = self.copy_population()

        new_population = []

        # copy last best 10% of population
        for i in new_population_index[:int(self.n/10)]:
            new_population.append(copied_population[i])

        # mutate best 50% of population
        copied_population = self.copy_population()
        for i in new_population_index[:int(self.n*5/10)]:
            new_population.append(copied_population[i])
            if np.random.random(1) > 0.2:
                new_population[-1].mutate_weight()
            if np.random.random(1) > 0.95:
                new_population[-1].mutate_connection()
            if np.random.random(1) > 0.97:
                new_population[-1].mutate_neuron()
            if np.random.random(1) > 0.97:
                new_population[-1].mutate_activation()

        # cross best 40% of population
        copied_population = self.copy_population()
        for i in new_population_index[:int(self.n*4/10)]:
            new_random = np.random.choice(new_population_index[:int(self.n*4/10)], 1)[0]
            new_population.append(copied_population[i].cross(copied_population[new_random]))

        self.population = new_population

    def copy_population(self):
        new_population = []
        for pop in self.population:
            new_population.append(Network(copy.deepcopy(pop.neuron_ids), copy.deepcopy(pop.connection_ids), self.neurons, self.connections, self.score_function, copy.deepcopy(pop.connection_data)))
        return new_population
