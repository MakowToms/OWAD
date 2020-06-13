from genetic.neuron import NeuronContainer
from genetic.connection import ConnectionContainer
from genetic.network import Network
import numpy as np
import copy


class NEAT:
    def __init__(self, n_input, n_output, n, score_function, data, speciation_threshold, epochs_to_remove_species=15, best_population_type="score", use_speciation=True):
        self.score_function = score_function
        self.neurons = NeuronContainer(n_input, n_output)
        self.connections = ConnectionContainer(self.neurons.inputs, self.neurons.outputs)
        self.neurons.set_connection_container(self.connections)
        self.n = n
        self.population = self.create_population()
        self.data = data
        self.speciation = None
        self.c1 = 1
        self.c2 = 1
        self.c3 = 0.4
        self.speciation_threshold = speciation_threshold
        self.speciation_index = 0
        self.cross_percentage = 0.75
        self.species_examples = None
        self.best_species_people = dict()
        self.banned_species = set()
        self.epochs_to_remove_species = epochs_to_remove_species
        self.best_population_type = best_population_type
        self.use_speciation = use_speciation
        self.best_scores = []
        self.best_accuracies = []

    def get_sequence_speciation(self):
        self.speciation_index += 1
        return self.speciation_index

    def create_population(self):
        population = []
        for i in range(self.n):
            population.append(Network(self.neurons.base_neurons, self.connections.base_connections, self.neurons, self.connections, self.score_function))
        return population

    def get_best_population(self, best_population_type="score"):
        results = np.zeros([len(self.population)])
        for index, pop in enumerate(self.population):
            results[index] = pop.score(copy.deepcopy(self.data), best_population_type)
        print(f'Best score: {results.max():0.03}')
        best_pop = results.argsort()[::-1]
        self.best_scores.append(results.max())
        accuracy = self.population[best_pop[0]].score(copy.deepcopy(self.data), "accuracy")
        print(f'Best accuracy: {accuracy}')
        self.best_accuracies.append(accuracy)
        return best_pop

    def compute_speciation(self, one, other):
        same_neurons = one.neuron_ids.intersection(other.neuron_ids)
        same_connections = one.connection_ids.intersection(other.connection_ids)
        excess_neurons = one.neuron_ids.difference(other.neuron_ids)
        excess_connections = one.connection_ids.difference(other.connection_ids)
        disjoint_neurons = other.neuron_ids.difference(one.neuron_ids)
        disjoint_connections = other.connection_ids.difference(one.connection_ids)
        E = len(excess_connections) + len(excess_neurons)
        D = len(disjoint_connections) + len(disjoint_neurons)
        N = max(len(one.neuron_ids) + len(one.connection_ids), len(other.neuron_ids) + len(other.connection_ids))
        W = 0
        for connection_id in same_connections:
            W += np.abs(one.connection_data[connection_id].weight - other.connection_data[connection_id].weight)
        W /= len(same_connections)
        delta = self.c1 * E / N + self.c2 * D / N + self.c3 * W
        return delta

    def divide_species(self):
        if self.species_examples is None:
            species = np.unique(self.speciation)
            example_species = {}
            for sp in species:
                this_species_indexes = np.arange(0, len(self.population))[self.speciation == sp]
                random_index = np.random.choice(this_species_indexes, 1)[0]
                random_person = self.copy_person(self.population[random_index])
                example_species[sp] = random_person
            self.species_examples = example_species
        for index, person in enumerate(self.population):
            for key, example in self.species_examples.items():
                if self.compute_speciation(person, example) < self.speciation_threshold:
                    self.speciation[index] = key
                    break
            else:
                new_index = self.get_sequence_speciation()
                self.speciation[index] = new_index
                self.species_examples[new_index] = self.copy_person(person)

    def cross_population(self):
        if self.use_speciation:
            for index, person in enumerate(self.population):
                if np.random.random(1) < self.cross_percentage:
                    speciation_number = self.speciation[index]
                    new_random = np.random.choice(np.arange(0, len(self.population))[self.speciation == speciation_number], 1)[0]
                    self.population.append(self.copy_person(person).cross(self.copy_person(self.population[new_random])))
                    self.speciation = np.hstack([self.speciation, speciation_number])
        else:
            for index, person in enumerate(self.population):
                if np.random.random(1) < self.cross_percentage:
                    new_random = np.random.choice(np.arange(0, len(self.population)), 1)[0]
                    self.population.append(self.copy_person(person).cross(self.copy_person(self.population[new_random])))

    def mutate_population(self):
        for index, person in enumerate(self.population):
            if np.random.random(1) > 0.2:
                person.mutate_weight()
            if np.random.random(1) > 0.95:
                person.mutate_connection()
            if np.random.random(1) > 0.97:
                person.mutate_neuron()
            if np.random.random(1) > 0.97:
                person.mutate_activation()

    def save_best_people_in_each_species(self, epoch_number):
        species = np.unique(self.speciation)
        for species_index in species:
            this_species_indexes = np.arange(0, len(self.population))[species_index == self.speciation]
            results = np.zeros([this_species_indexes.shape[0]])
            for i, index in enumerate(this_species_indexes):
                results[i] = self.population[index].score(copy.deepcopy(self.data), self.best_population_type)
            best = results.argsort()[::-1]
            if not self.best_species_people.__contains__(species_index):
                print(f"Creating species: {species_index} with best: {results[best[0]]}")
                self.best_species_people[species_index] = results[best[0]], self.copy_person(self.population[this_species_indexes[best[0]]]), epoch_number
            else:
                old_best, old_person, old_epoch = self.best_species_people[species_index]
                if results[best[0]] > old_best:
                    self.best_species_people[species_index] = results[best[0]], self.copy_person(self.population[this_species_indexes[best[0]]]), epoch_number
                elif old_epoch + self.epochs_to_remove_species < epoch_number:
                    self.banned_species.add(species_index)

    def epoch(self, epoch_number):
        if self.use_speciation:
            if self.speciation is None:
                self.speciation = np.zeros([self.n], dtype=np.int)
            else:
                self.divide_species()

        self.cross_population()
        self.mutate_population()

        if self.use_speciation:
            self.save_best_people_in_each_species(epoch_number)

        new_population_index = self.get_best_population(self.best_population_type)
        new_population = []

        if self.use_speciation:
            new_speciation = np.array([], dtype=np.int)
            species = np.unique(self.speciation)
            for species_index in species:
                if self.speciation[species_index == self.speciation].shape[0] >= 5:
                    new_population.append(self.copy_person(self.best_species_people[species_index][1]))
                    new_speciation = np.hstack([new_speciation, np.array([species_index])])

        index = 0
        while len(new_population) < self.n:
            if index >= len(self.population):
                index = 0
                print("\n\n\n Second time getting population \n\n\n")
            person_index = new_population_index[index]
            if self.use_speciation:
                if not self.banned_species.__contains__(self.speciation[person_index]):
                    new_population.append(self.copy_person(self.population[person_index]))
                    new_speciation = np.hstack([new_speciation, np.array([self.speciation[person_index]])])
            else:
                new_population.append(self.copy_person(self.population[person_index]))
            index += 1
        self.population = new_population
        if self.use_speciation:
            self.speciation = self.speciation[new_population_index[:self.n]]

    def copy_population(self):
        new_population = []
        for person in self.population:
            new_population.append(self.copy_person(person))
        return new_population

    def copy_person(self, person):
        return Network(copy.deepcopy(person.neuron_ids), copy.deepcopy(person.connection_ids), self.neurons,
                       self.connections, self.score_function, copy.deepcopy(person.connection_data))

