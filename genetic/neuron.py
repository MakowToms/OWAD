import enum


class NeuronContainer:
    def __init__(self, n_input, n_output):
        self.possible_neurons = {}
        self.base_neurons = set()
        self.current_id = -1
        self.inputs = self.initialize(n_input, 'input')
        self.outputs = self.initialize(n_output, 'output')

    def set_connection_container(self, connection_container):
        self.connection_container = connection_container

    def initialize(self, number, type):
        data = {}
        for i in range(number):
            new_id = self.get_sequence()
            new_neuron = Neuron(new_id, type)
            data[new_id] = new_neuron
            self.possible_neurons[new_id] = new_neuron
            self.base_neurons.add(new_id)
        return data

    def get_sequence(self):
        self.current_id += 1
        return self.current_id

    def create_new_neuron(self, type):
        new_id = self.get_sequence()
        new_neuron = Neuron(new_id, type)
        self.possible_neurons[new_id] = new_neuron
        return new_id

    def create_new_neuron_on_connection(self, connection_id, neuron_ids):
        connection = self.connection_container.possible_connections[connection_id]
        from_node = connection.from_node
        to_node = connection.to_node
        connections_from = self.connection_container.get_by_from_node(from_node)
        connections_to = self.connection_container.get_by_to_node(to_node)
        unique_endings_of_connections = {v for k, v in connections_from.items()}
        unique_begins_of_connections = {v for k, v in connections_to.items()}
        both = unique_begins_of_connections.intersection(unique_endings_of_connections)
        for both_elem in both:
            if not neuron_ids.__contains__(both_elem):
                # it means that it don't have to create new neuron, but use this neuron
                connection_to_both = [k for k, v in connections_from.items() if v == both_elem][0]
                connection_from_both = [k for k, v in connections_to.items() if v == both_elem][0]
                return both_elem, connection_to_both, connection_from_both
        # create neuron and create new connections
        new_neuron = self.create_new_neuron(NeuronType.hidden)
        connection_to_new = self.connection_container\
            .create_new_connection_between_nodes_if_not_exist(from_node, new_neuron)
        connection_from_new = self.connection_container\
            .create_new_connection_between_nodes_if_not_exist(new_neuron, to_node)
        return new_neuron, connection_to_new, connection_from_new


class Neuron:
    def __init__(self, id, type):
        self.id = id
        self.type = NeuronType.get_neuron_type(type)


class NeuronType(enum.Enum):
    input = 1
    hidden = 2
    output = 3

    @staticmethod
    def get_neuron_type(name):
        if name in NeuronType.__members__:
            return NeuronType[name]
        else:
            return NeuronType.hidden

