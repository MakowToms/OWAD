import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class Kohonen:
    def __init__(self, M, N, dimension, neighbour_param=1, lambd=1, distance_type="gauss", grid_type="square", t=1,
                 use_pca=False):
        self.weights = np.random.random([M, N, dimension]) - np.ones([M, N, dimension]) / 2
        self.grid_type = grid_type
        self.create_grid(M, N)
        self.neighbour_param = neighbour_param
        self.t = t
        self.lambd = lambd
        self.distance_type = distance_type
        self.x = N
        self.y = M
        self.shape = (M, N)
        self.distances = self.compute_distances_grid(M, N)
        self.use_pca = use_pca
        if use_pca:
            self.pca = PCA(n_components=2)

    # points on the grid
    def create_grid(self, M, N):
        if self.grid_type == "square":
            self.create_square_grid(M, N)
        else:
            self.create_hexagonal_grid(M, N)

    def create_square_grid(self, M, N):
        x = np.linspace(0, N - 1, N)
        y = np.linspace(0, M - 1, M)
        grid_x, grid_y = np.meshgrid(x, y)
        self.grid_x = grid_x
        self.grid_y = grid_y

    def create_hexagonal_grid(self, M, N):
        """
        it creates a cube grid of coordinates -
        the easiest to compute distances between points
        :param M:
        :param N:
        :return:
        """
        grid_x = np.zeros([M, N])
        grid_y = np.zeros([M, N])
        grid_z = np.zeros([M, N])
        coords = (0, 0, 0)
        for i in range(M):
            if i == 0:
                pass
            elif i % 2 == 0:
                coords = (coords[0] - 1, coords[1], coords[2] + 1)
            else:
                coords = (coords[0], coords[1] - 1, coords[2] + 1)
            this_coords = coords
            for j in range(N):
                this_coords = (this_coords[0] + 1, this_coords[1] - 1, this_coords[2])
                grid_x[i, j] = this_coords[0]
                grid_y[i, j] = this_coords[1]
                grid_z[i, j] = this_coords[2]
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z

    # finding nearest point in terms of weights
    def find_nearest_point(self, point):
        distances = self.weights - point
        distance_matrix = np.zeros(self.weights.shape[:2])
        for i in range(self.weights.shape[2]):
            distance_matrix += (distances[:, :, i] ** 2)
        distance_matrix = np.sqrt(distance_matrix)
        return np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)

    # function to compute distances between every point and store it -
    # then it needs only to get proper values to find distance between points on the grid
    # (no need to compute it many times)
    def compute_distances_grid(self, M, N):
        distances = []
        for i in range(M):
            this_distances = []
            for j in range(N):
                this_distances.append(self.compute_distance_between_point([i, j]))
            distances.append(this_distances)
        return distances

    def compute_distance_between_point(self, index_point):
        if self.grid_type == "square":
            return self.distance_on_the_square_grid(index_point)
        else:
            return self.distance_on_the_hexagonal_grid(index_point)

    def distance_on_the_square_grid(self, index_point):
        grid_x = self.grid_x
        grid_y = self.grid_y
        # manhattan distance
        distances = np.absolute(grid_y - index_point[0]) + np.absolute(grid_x - index_point[1])
        # euclidean distance
        # distances = np.sqrt((grid_y - index_point[0]) ** 2 + (grid_x - index_point[1]) ** 2)
        return distances

    # function to compute distances between point and other points on hexagonal grid
    def distance_on_the_hexagonal_grid(self, index_point):
        distances = np.maximum(np.absolute(self.grid_x - self.grid_x[index_point[0], index_point[1]]),
                               np.absolute(self.grid_y - self.grid_y[index_point[0], index_point[1]]),
                               np.absolute(self.grid_z - self.grid_z[index_point[0], index_point[1]]))
        return distances

    # distance in terms of grid
    def distance_on_the_grid(self, index_point):
        return self.distances[index_point[0]][index_point[1]]

    def gaussian_distance(self, index_point):
        distances = self.distance_on_the_grid(index_point)
        return np.exp(- ((distances * self.neighbour_param * self.t) ** 2))

    def mexican_hat_distance(self, index_point):
        distances = self.distance_on_the_grid(index_point)
        return (2 - 4 * ((distances * self.neighbour_param * self.t) ** 2)) * \
               np.exp(- ((distances * self.neighbour_param * self.t) ** 2))

    def alpha_t(self):
        return np.exp(- self.t / self.lambd)

    # update weights for single point
    def update_weights(self, point, index_point):
        if self.distance_type == "gauss":
            distances = self.gaussian_distance(index_point)
        else:
            distances = self.mexican_hat_distance(index_point)
        weights_shape = self.weights.shape
        distances = distances.reshape([weights_shape[0], weights_shape[1], 1])
        distances = np.repeat(distances, weights_shape[2], axis=2)
        self.weights += distances * self.alpha_t() * (point - self.weights)

    # learn one epoch
    def learn_epoch(self, x):
        x = self.__shuffle__(x)
        for i in range(x.shape[0]):
            point = x[i]
            index_point = self.find_nearest_point(point)
            self.update_weights(point, index_point)

    # learn many epochs with changing some parameters during learning
    def learn_epochs(self, x, epochs=10):
        if self.use_pca:
            self.pca.fit(x)
        self.lambd = epochs
        for i in range(1, epochs + 1):
            self.t = i
            self.learn_epoch(x)

    # plot weights of the network in 2d
    def plot_weights(self):
        weights = self.weights
        if self.use_pca:
            weights = weights.reshape([self.y * self.x, weights.shape[2]])
            weights = self.pca.transform(weights).reshape([self.y, self.x, 2])
        if self.grid_type == "square":
            self.plot_weights_square_grid(weights)
        else:
            self.plot_weights_hexagonal_grid(weights)

    def plot_weights_square_grid(self, weights):
        for i in range(self.y):
            plt.plot(weights[i, :, 0], weights[i, :, 1], "k")
        for i in range(self.x):
            plt.plot(weights[:, i, 0], weights[:, i, 1], "k")

    def plot_weights_hexagonal_grid(self, weights):
        # horizontal line
        for i in range(self.y):
            plt.plot(weights[i, :, 0], weights[i, :, 1], "k")
        # 2 diagonal lines (plot together)
        for i in range(self.y - 1):
            if i % 2 == 0:
                plt.plot(weights[i:(i + 2), :, 0].reshape((-1, 1), order='F'),
                         weights[i:(i + 2), :, 1].reshape((-1, 1), order='F'), "k")
            else:
                plt.plot(weights[i + 1:(i - 1):-1, :, 0].reshape((-1, 1), order='F'),
                         weights[i + 1:(i - 1):-1, :, 1].reshape((-1, 1), order='F'), "k")

    def transform_with_pca(self, x):
        if self.use_pca:
            return self.pca.transform(x)
        else:
            return x

    @staticmethod
    def __shuffle__(x):
        s = np.arange(x.shape[0])
        np.random.shuffle(s)
        return x[s]
