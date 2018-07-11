import numpy as np
import matplotlib.pyplot as plt

from HelperFunctions.HelperFunctions import plot_winner_board

# Folder Paths
data_load_folder = 'C:/Users/Jordan Yeomans/Documents/GitHub/RiddlesIO/four_in_a_row/Data/Processed_Data/Random_vs_Random/'

# Load Data
move_history_input = np.load(data_load_folder + 'input_data.npy')
move_history_output = np.load(data_load_folder + 'output_data.npy')

print('Input Shape = {}'.format(move_history_input.shape))
print('Output Shape = {}'.format(move_history_output.shape))

plot_range = np.arange(10, 30, 1)

for i in plot_range:
    print('New Board')
    print('Next Move = {}'.format(np.argmax(move_history_output[i])))
    plot_winner_board(move_history_input[i])