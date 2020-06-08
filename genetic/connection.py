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

    def create_new_connection_between_nodes_if_not_exist(self, from_node, to_node):
        existing = self.get_existing_connection(from_node, to_node)
        if existing is None:
            return self.create_new_connection(from_node, to_node)
        else:
            return existing

    def get_connection_by_id(self, id):
        return self.possible_connections[id]

    def get_by_from_node(self, from_node):
        return {k: v.to_node for k, v in self.possible_connections.items() if v.from_node == from_node}

    def get_by_to_node(self, to_node):
        return {k: v.from_node for k, v in self.possible_connections.items() if v.to_node == to_node}

    def get_existing_connection(self, from_node, to_node, connection_ids=None):
        if connection_ids is None:
            existing = [k for k, v in self.possible_connections.items() if (v.to_node==to_node and v.from_node==from_node)]
        else:
            existing = [k for k, v in self.possible_connections.items() if (v.to_node==to_node and v.from_node==from_node and connection_ids.__contains__(k))]
        if len(existing) > 0:
            if len(existing) > 1:
                Warning("At least two connections between two nodes")
            return existing[0]
        return None

    def exist_connection_in_connection_ids(self, from_node, to_node, connection_ids):
        return self.get_existing_connection(from_node, to_node, connection_ids) is not None


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
