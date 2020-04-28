'''
This code is copied from: https://github.com/alexarnimueller/som
The copied code is under MIT license:
MIT License

Copyright (c) 2018 Alex Müller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mptchs


def plot_point_map(data, targets, kohonen, targetnames, filename=None, colors=None, markers=None, example_dict=None,
                   density=True, activities=None):
    """ Visualize the som with all data as points around the neurons
    :param data: {numpy.ndarray} data to visualize with the SOM
    :param targets: {list/array} array of target classes (0 to len(targetnames)) corresponding to data
    :param targetnames: {list/array} names describing the target classes given in targets
    :param filename: {str} optional, if given, the plot is saved to this location
    :param colors: {list/array} optional, if given, different classes are colored in these colors
    :param markers: {list/array} optional, if given, different classes are visualized with these markers
    :param example_dict: {dict} dictionary containing names of examples as keys and corresponding descriptor values
        as values. These examples will be mapped onto the density map and marked
    :param density: {bool} whether to plot the density map with winner neuron counts in the background
    :param activities: {list/array} list of activities (e.g. IC50 values) to use for coloring the points
        accordingly; high values will appear in blue, low values in green
    :return: plot shown or saved if a filename is given
    """
    if not markers:
        markers = ['o'] * len(targetnames)
    if not colors:
        colors = ['#EDB233', '#90C3EC', '#C02942', '#79BD9A', '#774F38', 'gray', 'black']
    if activities:
        heatmap = plt.get_cmap('coolwarm').reversed()
        colors = [heatmap(a / max(activities)) for a in activities]
    if density:
        fig, ax = plot_density_map(data, kohonen, internal=True)
    else:
        fig, ax = plt.subplots(figsize=kohonen.shape)

    for cnt, xx in enumerate(data):
        if activities:
            c = colors[cnt]
        else:
            c = colors[targets[cnt]]
        w = kohonen.find_nearest_point(xx)
        ax.plot(w[1] + .5 + 0.1 * np.random.randn(1), w[0] + .5 + 0.1 * np.random.randn(1),
                markers[targets[cnt]], color=c, markersize=12)

    ax.set_aspect('equal')
    ax.set_xlim([0, kohonen.x])
    ax.set_ylim([0, kohonen.y])
    plt.xticks(np.arange(.5, kohonen.x + .5), range(kohonen.x))
    plt.yticks(np.arange(.5, kohonen.y + .5), range(kohonen.y))
    ax.grid(which='both')

    if not activities:
        patches = [mptchs.Patch(color=colors[i], label=targetnames[i]) for i in range(len(targetnames))]
        legend = plt.legend(handles=patches, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(targetnames),
                            mode="expand", borderaxespad=0.1)
        legend.get_frame().set_facecolor('#e5e5e5')

    if example_dict:
        for k, v in example_dict.items():
            w = kohonen.find_nearest_point(v)
            x = w[1] + 0.5 + np.random.normal(0, 0.15)
            y = w[0] + 0.5 + np.random.normal(0, 0.15)
            plt.plot(x, y, marker='*', color='#FDBC1C', markersize=24)
            plt.annotate(k, xy=(x + 0.5, y - 0.18), textcoords='data', fontsize=18, fontweight='bold')

    if filename:
        plt.savefig(filename)
        plt.close()
        print("Point map plot done!")
    else:
        plt.show()


def plot_density_map(data, kohonen, colormap='Oranges', filename=None, example_dict=None, internal=False):
    """ Visualize the data density in different areas of the SOM.
    :param data: {numpy.ndarray} data to visualize the SOM density (number of times a neuron was winner)
    :param colormap: {str} colormap to use, select from matplolib sequential colormaps
    :param filename: {str} optional, if given, the plot is saved to this location
    :param example_dict: {dict} dictionary containing names of examples as keys and corresponding descriptor values
        as values. These examples will be mapped onto the density map and marked
    :param internal: {bool} if True, the current plot will stay open to be used for other plot functions
    :return: plot shown or saved if a filename is given
    """
    wm = winner_map(data, kohonen)
    fig, ax = plt.subplots(figsize=[kohonen.shape[1], kohonen.shape[0]])
    plt.pcolormesh(wm, cmap=colormap, edgecolors=None)
    plt.colorbar()
    plt.xticks(np.arange(.5, kohonen.x + .5), range(kohonen.x))
    plt.yticks(np.arange(.5, kohonen.y + .5), range(kohonen.y))
    ax.set_aspect('equal')

    if example_dict:
        for k, v in example_dict.items():
            w = kohonen.find_nearest_point(v)
            x = w[1] + 0.5 + np.random.normal(0, 0.15)
            y = w[0] + 0.5 + np.random.normal(0, 0.15)
            plt.plot(x, y, marker='*', color='#FDBC1C', markersize=24)
            plt.annotate(k, xy=(x + 0.5, y - 0.18), textcoords='data', fontsize=18, fontweight='bold')

    if not internal:
        if filename:
            plt.savefig(filename)
            plt.close()
            print("Density map plot done!")
        else:
            plt.show()
    else:
        return fig, ax


def winner_map(data, kohonen):
    """ Get the number of times, a certain neuron in the trained SOM is winner for the given data.
    :param data: {numpy.ndarray} data to compute the winner neurons on
    :return: {numpy.ndarray} map with winner counts at corresponding neuron location
    """
    wm = np.zeros(kohonen.shape, dtype=int)
    for d in data:
        [x, y] = kohonen.find_nearest_point(d)
        wm[x, y] += 1
    return wm



