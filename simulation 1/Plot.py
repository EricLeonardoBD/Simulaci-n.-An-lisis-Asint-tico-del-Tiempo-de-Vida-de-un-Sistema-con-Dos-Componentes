import numpy as np
import matplotlib.pyplot as plt
from math import e

def plot_data_and_functions(filename, func1, func2, x_range):
    """
    Reads data from a file, plots a scatter plot of the 2nd and 8th columns,
    and plots two functions.

    Args:
        filename: The name of the file containing the data table.
                 The first line of the file must be the number of rows.
        func1: A function that takes a single argument (x) and returns a y-value.
        func2: A function that takes a single argument (x) and returns a y-value.
        x_range: A tuple (x_min, x_max) specifying the range of x values for
                 plotting the functions.
    """

    try:
        with open(filename, 'r') as f:
            num_rows = int(f.readline().strip())  # Read and parse number of rows

            # Read the data table (skip the first line which is #rows).
            data = np.loadtxt(f, max_rows=num_rows)

            # Check if the file actually had #rows rows in it.
            if data.shape[0] != num_rows:
              print(f"Warning: File specifies {num_rows} rows, but only found {data.shape[0]} rows.")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return
    except ValueError:
        print(f"Error: Invalid data in file '{filename}'. Ensure the file format is correct.")
        return


    # Extract x and y coordinates from the data table (2nd and 8th columns)
    x_coords = data[:, 1]  # Second column (index 1)
    y_coords = data[:, 7]  # Eighth column (index 7)

    # Create the scatter plot
    plt.figure(figsize=(8, 6))  # Adjust figure size if needed
    plt.scatter(x_coords, y_coords, label="Reparaciones " + filename.split()[0], marker='o', c = "tab:blue")

    # Generate x values for plotting the functions
    x_vals = np.linspace(x_range[0], x_range[1], 100)  # 100 points
    y_vals_func1 = [func1(x) for x in x_vals] #Use list comprehension to create the array.
    y_vals_func2 = [func2(x) for x in x_vals]

    # Plot the functions
    plt.plot(x_vals, y_vals_func1, label="Reparaciones exponenciales", linestyle='--', c = "pink")
    plt.plot(x_vals, y_vals_func2, label="Reparaciones constantes", linestyle='--', c = "lightgreen")

    # Add labels, title, and legend
    plt.xlabel("r = EX / EY")
    plt.ylabel("Error Relativo")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


if __name__ == '__main__':
    
    filename = "Gamma results.txt"  

    # Define other cases functions 
    def my_func1(x):
        return 1 - (x**2*(2*x + 1)/ (2*x*(x**2 + 6*x + 3)))

    def my_func2(x):
        return 1 - ((1 + e**(-1/x) - 2*e**(-2/x))/(3/x * (1.5 + 1.5*e**(-1/x) - 2*e**(-2/x))))

    # Define the x-range for plotting the functions
    x_range = (0.1, 60)  # x values from 0.1 to 60

    plot_data_and_functions(filename, my_func1, my_func2, x_range)
