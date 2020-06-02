import numpy as np


class ConnectionContainer:
    def __init__(self, inputs, outputs):
        self.possible_connections = {}
        self.current_id = -1
        self.base_connections = set()
        self.create_base_connections(inputs, outputs)

    def create_base_connections(self, inputs, outputs):
        for input in inputs:
            for output in outputs:
                self.base_connections.add(self.create_new_connection(input, output))

    def get_sequence(self):
        self.current_id += 1
        return self.current_id

    def create_new_connection(self, from_node, to_node):
        new_id = self.get_sequence()
        new_connection = Connection(from_node, to_node, new_id)
        self.possible_connections[new_id] = new_connection
        return new_id

    def get_connection_by_id(self, id):
        return self.possible_connections[id]


class Connection:
    def __init__(self, from_node, to_node, id):
        self.from_node = from_node
        self.to_node = to_node
        self.id = id
        self.weight = self.generate_new_weight()
        self.active = True

    @staticmethod
    def generate_new_weight():
        return np.random.standard_normal(1)
