import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(data, classes, output, **kwargs):
    """
    Plots a confusion matrix using seaborn's heatmap.

    This function creates a confusion matrix plot with seaborn's heatmap visualization,
    labeling the axes with the provided class names. The plot is then saved to the specified
    output file path and the plot window is closed to free up resources. It now supports
    additional customization through keyword arguments.

    Args:
        data (array-like): The confusion matrix data. Must be a 2D array where the element at
                           [i, j] represents the number of instances of class i predicted as class j.
        classes (list of str): The names of the classes corresponding to the indices of the confusion matrix.
        output (str): The file path where the confusion matrix plot will be saved.
        **kwargs: Additional keyword arguments for plot customization. Supported keywords:

            - figsize (tuple): Figure size in inches (width, height). Defaults to (10, 10).
            - labels (tuple): A tuple of strings for the x and y axis labels. Defaults to ('Predicted', 'True').

    Returns:
        None
    """
    figsize = kwargs.get('figsize', (10, 10))
    xlabel, ylabel = kwargs.get('labels', ('Predicted', 'True'))

    plt.figure(figsize=figsize)
    sns.heatmap(data, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Confusion Matrix')
    plt.savefig(output)
    plt.close()
