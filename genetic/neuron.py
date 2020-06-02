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

    def create_new_neuron_on_connection(self, connection_id):
        connection = self.connection_container[connection_id]
        return self.create_new_neuron(NeuronType.hidden)


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

