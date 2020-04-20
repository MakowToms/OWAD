import numpy as np

M = 3
N = 3
dimension = 2
my_map = np.random.random([M, N, dimension]) - np.ones([M, N, dimension]) / 2
point = np.array([-1, 1])
index_point = np.array([1, 1])
my_map - point
distance_matrix = np.linalg.norm(my_map - point, axis=2)
np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)


class Kohonen:
    def __init__(self, M, N, dimension, neighbour_param=1, lambd=1, distance_type="gauss", t=1):
        self.weights = np.random.random([M, N, dimension]) - np.ones([M, N, dimension]) / 2
        self.create_grid(M, N)
        self.neighbour_param = neighbour_param
        self.t = t
        self.lambd = lambd
        self.distance_type = distance_type
        self.x = N
        self.y = M
        self.shape = (M, N)

    def create_grid(self, M, N):
        x = np.linspace(0, N-1, N)
        y = np.linspace(0, M-1, M)
        grid_x, grid_y = np.meshgrid(x, y)
        self.grid_x = grid_x
        self.grid_y = grid_y

    def find_nearest_point(self, point):
        distance_matrix = np.linalg.norm(self.weights - point, axis=2)
        return np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)

    def distance_on_the_grid(self, index_point):
        grid_x = self.grid_x
        grid_y = self.grid_y
        distances = np.sqrt((grid_x - index_point[0]) ** 2 + (grid_y - index_point[1]) ** 2)
        return distances

    def gaussian_distance(self, index_point):
        distances = self.distance_on_the_grid(index_point)
        return np.exp(- ((distances * self.neighbour_param) ** 2))

    def mexican_hat_distance(self, index_point):
        distances = self.distance_on_the_grid(index_point)
        return (2 - 4 * ((distances * self.neighbour_param) ** 2)) *\
               np.exp(- ((distances * self.neighbour_param) ** 2))

    def alpha_t(self):
        return np.exp(- self.t / self.lambd)

    def update_weights(self, point, index_point):
        if self.distance_type == "gauss":
            distances = self.gaussian_distance(index_point)
        else:
            distances = self.mexican_hat_distance(index_point)
        weights_shape = self.weights.shape
        distances = distances.reshape([weights_shape [0], weights_shape [1], 1])
        distances = np.repeat(distances, weights_shape[2], axis=2)
        self.weights += distances * self.alpha_t() * (point - self.weights)

    def learn_epoch(self, x):
        x = self.__shuffle__(x)
        for i in range(x.shape[0]):
            point = x[i]
            index_point = self.find_nearest_point(point)
            self.update_weights(point, index_point)

    def learn_epochs(self, x, epochs=10):
        for i in range(1, epochs+1):
            self.t = i
            self.learn_epoch(x)

    @staticmethod
    def __shuffle__(x):
        s = np.arange(x.shape[0])
        np.random.shuffle(s)
        return x[s]
