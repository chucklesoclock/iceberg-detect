import os
import matplotlib.pyplot as plt


def save_graph_in_folder(folder_name, file_name):
    try:
        plt.savefig(os.path.join(folder_name, file_name),
                    frameon=True, bbox_inches='tight')
    except FileNotFoundError:
        os.mkdir(folder_name)
        plt.savefig(os.path.join(folder_name, file_name),
                    frameon=True, bbox_inches='tight')
