from genetic.neuron import NeuronContainer
from genetic.connection import ConnectionContainer
import copy
import numpy as np
from neural_network.activations import sigmoid


class Network:
    def __init__(self, neuron_ids, connection_ids, neurons_container, connection_container, score_function, connection_data=None):
        self.score_function = score_function
        self.neuron_ids = copy.deepcopy(neuron_ids)
        self.connection_ids = copy.deepcopy(connection_ids)
        self.neurons_container = neurons_container
        self.connection_container = connection_container
        if connection_data is None:
            connection_data = self.initialize_connections()
        self.connection_data = connection_data

    def evaluate(self, data):
        done_neurons = {}
        used_connections = set()
        neurons_to_propagate = {}
        connection_ids = self.get_active_connections()
        for id in data.keys():
            neurons_to_propagate[id] = data[id]
        while len(used_connections) < len(connection_ids):
            new_neurons_to_propagate = {}
            for connection_id in connection_ids:
                connection = self.connection_container.get_connection_by_id(connection_id)
                if connection.from_node in neurons_to_propagate.keys():
                    new_value = sigmoid(neurons_to_propagate[connection.from_node]) * self.connection_data[connection_id].weight
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
        try:
            result = self.score_function(self.evaluate(data))
        except:
            print("\n\nNot suceed\n\n")
            result = 0
        return result

    def initialize_connections(self):
        connection_data = {}
        for connection_id in self.connection_ids:
            connection_data[connection_id] = ConnectionData()
        return connection_data

    def get_active_connections(self):
        connections = set()
        for con in self.connection_ids:
            if self.connection_data[con].active:
                connections.add(con)
        return connections

    def mutate_weight(self):
        for connection_id in self.connection_data:
            self.connection_data[connection_id].mutate()

    def mutate_neuron(self):
        connection_id = np.random.choice(list(self.connection_ids), 1)[0]
        new_neuron_id, first_con_id, second_con_id = self.neurons_container.create_new_neuron_on_connection(connection_id, self.neuron_ids)
        self.neuron_ids.add(new_neuron_id)
        self.connection_ids.remove(connection_id)
        self.connection_ids.add(first_con_id)
        self.connection_ids.add(second_con_id)
        self.connection_data[first_con_id] = ConnectionData(self.connection_data[connection_id].weight)
        self.connection_data[second_con_id] = ConnectionData(1)
        self.connection_data.pop(connection_id)

    def mutate_connection(self):
        nodes = self.get_random_possible_new_connection()
        if nodes is None:
            return
        connection_id = self.connection_container\
            .create_new_connection_between_nodes_if_not_exist(nodes[0], nodes[1])
        self.connection_ids.add(connection_id)
        self.connection_data[connection_id] = ConnectionData()

    def mutate_activation(self):
        self.connection_data[np.random.choice(list(self.connection_ids), 1)[0]].mutate_activation()

    def get_random_possible_new_connection(self):
        nodes = None
        tries = 0
        while nodes is None and tries < 100:
            nodes = np.random.choice(list(self.neuron_ids), 2)
            if self.connection_container.exist_connection_in_connection_ids(nodes[0], nodes[1], self.connection_ids):
                nodes = None
                tries += 1
        return nodes

    def cross(self, other):
        neuron_ids = self.neuron_ids.union(other.neuron_ids)
        connection_ids = self.connection_ids.union(other.connection_ids)
        connection_data = copy.deepcopy(self.connection_data)
        for key, value in other.connection_data.items():
            if connection_data.__contains__(key):
                if np.random.random(1) > 0.5:
                    connection_data[key] = value
            else:
                connection_data[key] = value
        return Network(neuron_ids, connection_ids, self.neurons_container,
                       self.connection_container, self.score_function, connection_data)


class ConnectionData:
    def __init__(self, weight=None):
        if weight is None:
            weight = self.generate_new_weight()
        self.weight = weight
        self.active = True

    def mutate(self):
        if np.random.randint(0, 9) == 1:
            self.weight = np.random.standard_normal(1)
        else:
            self.weight += np.random.standard_normal(1)

    def mutate_activation(self):
        self.active = not self.active

    @staticmethod
    def generate_new_weight():
        return np.random.standard_normal(1)
