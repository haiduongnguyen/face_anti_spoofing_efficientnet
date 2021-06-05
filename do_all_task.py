def show_image_size_survey():
    """
    task 1 : show result of survey image size
    """
    import os 
    import cv2
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import sys


    def show_graph(d:dict):
        """
        this function read data from a python dict
        then draw a graph to visualize data in dict
        """
        x = []
        y = []
        for key, value in d.items():
            x.append(str(key))
            y.append(value)

        x_pos = [i for i, _ in enumerate(x)]
        plt.figure()
        plt.bar(x_pos, y, color='green')
        plt.xlabel("Size")
        plt.ylabel("Number of images")
        plt.title("Count by size = width + height ")
        plt.xticks(x_pos, x)

    size_train= {100: 1077, 200: 17298, 300: 36400, 400: 42121, 500: 39558, 600: 34338, 700: 19799, 800: 33470}
    size_test = {100: 219, 200: 3483, 300: 6772, 400: 6289, 500: 7343, 600: 5800, 700: 4109, 800: 10378}

    show_graph(size_train)
    show_graph(size_test)
    plt.show()


if __name__ == '__main__':
    show_image_size_survey()