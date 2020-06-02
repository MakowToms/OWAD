from genetic.neuron import NeuronContainer
from genetic.connection import ConnectionContainer
import copy
import numpy as np


class Network:
    def __init__(self, neuron_ids, connection_ids, neurons_container, connection_container, score_function):
        self.score_function = score_function
        self.neuron_ids = copy.deepcopy(neuron_ids)
        self.connection_ids = copy.deepcopy(connection_ids)
        self.neurons_container = neurons_container
        self.connection_container = connection_container
        self.connection_data = self.initialize_connections()

    def evaluate(self, data):
        done_neurons = {}
        used_connections = set()
        neurons_to_propagate = {}
        for id in data.keys():
            neurons_to_propagate[id] = data[id]
        while len(used_connections) < len(self.connection_ids):
            new_neurons_to_propagate = {}
            for connection_id in self.connection_ids:
                connection = self.connection_container.get_connection_by_id(connection_id)
                if connection.from_node in neurons_to_propagate.keys():
                    new_value = neurons_to_propagate[connection.from_node] * self.connection_data[connection_id].weight
                    if new_neurons_to_propagate.__contains__(connection.to_node):
                        new_neurons_to_propagate[connection.to_node] += new_value
                    else:
                        new_neurons_to_propagate[connection.to_node] = new_value
                used_connections.add(connection_id)
            for neuron_id in neurons_to_propagate.keys():
                if done_neurons.__contains__(neuron_id):
                    done_neurons[neuron_id] += neurons_to_propagate[neuron_id]
                else:
                    done_neurons[neuron_id] = neurons_to_propagate[neuron_id]
            neurons_to_propagate = new_neurons_to_propagate
        for neuron_id in neurons_to_propagate.keys():
            if done_neurons.__contains__(neuron_id):
                done_neurons[neuron_id] += neurons_to_propagate[neuron_id]
            else:
                done_neurons[neuron_id] = neurons_to_propagate[neuron_id]
        return done_neurons

    def score(self, data):
        return self.score_function(self.evaluate(data))

    def initialize_connections(self):
        connection_data = {}
        for connection_id in self.connection_ids:
            connection_data[connection_id] = ConnectionData()
        return connection_data

    def mutate_weight(self):
        for connection_id in self.connection_data:
            self.connection_data[connection_id].mutate()

    def mutate_neuron(self):
        connection_id = np.random.choice(list(self.connection_ids), 1)
        new_neuron_id = self.neurons_container.create_new_neuron_on_connection(connection_id)
        self.neuron_ids.add(new_neuron_id)


class ConnectionData:
    def __init__(self):
        self.weight = self.generate_new_weight()
        self.active = True

    def mutate(self):
        if np.random.randint(0, 9)==1:
            self.weight = np.random.standard_normal(1)
        else:
            self.weight += np.random.standard_normal(1)

    @staticmethod
    def generate_new_weight():
        return np.random.standard_normal(1)
