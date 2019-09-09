import matplotlib.pyplot as plt

def myPlot(x, y):
    """
    Plots the data points x and y into a new figure. Plots the data
    points and gives the figure axes labels of population and profit.

    Parameters
    ----------
    x : array_like
        Data point values for x-axis.

    y : array_like
        Data point values for y-axis. Note x and y should have the same size.

    Instructions
    ------------
    Plot the training data into a figure using the "figure" and "plot"
    functions. Set the axes labels using the "xlabel" and "ylabel" functions.
    Assume the population and revenue data have been passed in as the x
    and y arguments of this function.

    Hint
    ----
    You can use the 'ro' option with plot to have the markers
    appear as red circles. Furthermore, you can make the markers larger by
    using plot(..., 'ro', ms=10), where `ms` refers to marker size. You
    can also set the marker edge color using the `mec` property.
    """
    #fig = pyplot.figure()  # open a new figure

    # ====================== YOUR CODE HERE =======================
    #plotData(x, y)

    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in 10,000s')
    #plt.legend('Training data')
    plt.plot(x, y, 'rx', label='Training Data')
    #plt.axis([0, 6, 0, 20])
    #plt.show()

    # =============================================================
