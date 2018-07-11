'''
AI Learns to Play Connect 4
Part 2 - Reading Saved Game Files
Jordan Yeomans - 2018
'''

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_board(board):
    ''' This function reads a saved board and plots Red and Yellow tokens in their correct positions

    Input: Numpy Array of Shape(6,7) containing:

    1 represents yellow tokens
    -1 represents red tokens
    0 represents an open space

    Returns:
    Shows a plot of the current board
    '''

    plt.figure()
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            if board[5-row][col] == 1:
                plt.scatter(col, row, c='Yellow', s=500, edgecolors='black')

            if board[5-row][col] == -1:
                plt.scatter(col, row, c='Red', s=500, edgecolors='black')

    plot_margin = 0.4                         # Padding around edges
    plt.grid()                                # Turn on grid
    plt.ylim(-plot_margin, 5 + plot_margin)   # Set Y Limits
    plt.xlim(-plot_margin, 6 + plot_margin)   # Set X Limits
    plt.show()                                # Show Plot

# Parameters for us to change
folder = './Games/'
game_num_to_plot = 0
round_to_plot = 9

games = os.listdir(folder)
current_game = np.load(folder + games[game_num_to_plot])

# Find last round
for round in range(current_game.shape[0]):
    current_game_abs = np.abs(current_game[round])
    current_game_sum = np.sum(current_game_abs)
    if current_game_sum == 0:
        round -= 1
        break

# Get final board
final_board = current_game[round]

# Plot the final board
plot_board(final_board)