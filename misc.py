import numpy as np
import hapke
import matplotlib.pyplot as plt
from scipy import optimize
from matplotlib.patches import Polygon

def plot_shaded_square(vertices):
    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Create a Polygon patch using the vertices
    square_patch = Polygon(vertices, closed=True, facecolor='yellow', alpha=0.4)

    # Add the square patch to the plot
    ax.add_patch(square_patch)

    # Set the aspect ratio to be equal to have a square plot
    ax.set_aspect('equal')

    # Set the axis limits based on the range of x and y values
    x, y = zip(*vertices)
    ax.set_xlim(min(x) - 1, max(x) + 1)
    ax.set_ylim(min(y) - 1, max(y) + 1)

    # Label the axes
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # Set the title of the plot
    ax.set_title('Plotting a Shaded Square')

    # Show the plot
    plt.show()

# Replace the 'vertices' variable with your four data points (x, y)
vertices = [(0, 1), (0, 5), (5, 5), (5, 0)]
plot_shaded_square(vertices)
