'''
AI Learns to Play Connect 4
Part 4 - Organise Data
Jordan Yeomans - 2018
'''

import numpy as np
import os

from HelperFunctions.HelperFunctions import find_last_round, find_winning_moves

# Parameters
total_games = 12000

# Variables
move_count = 0
total_dud_rounds = 0
move_history_input = np.zeros((total_games * 30, 6, 7))
move_history_output = np.zeros((total_games * 30, 7, 1))

red_load_folder = 'C:/Users/Jordan Yeomans/Documents/GitHub/RiddlesIO/four_in_a_row/Data/Raw_Data/Random_vs_Random/Red/'
yellow_load_folder = 'C:/Users/Jordan Yeomans/Documents/GitHub/RiddlesIO/four_in_a_row/Data/Raw_Data/Random_vs_Random/Yellow/'
save_folder = 'C:/Users/Jordan Yeomans/Documents/GitHub/RiddlesIO/four_in_a_row/Data/Processed_Data/Random_vs_Random/'

all_games_red = os.listdir(red_load_folder)
all_games_yellow = os.listdir(yellow_load_folder)

assert len(all_games_red) == len(all_games_yellow)

if total_games > len(all_games_red):
    print('Warning: Restricting total games from {} to {}'.format(total_games, len(all_games_red)))
    total_games = len(all_games_red)


# Iterate over the total number of games we want to store
for game_num in range(total_games):

    # Load Red and Yellow player's history
    board_red = np.load(red_load_folder + all_games_red[game_num])
    board_yellow = np.load(yellow_load_folder + all_games_yellow[game_num])

    # # Determine who played first
    # first = 'red' if np.sum(np.abs(board_red[0])) == 0 else 'yellow'

    # Find Longest Round
    last_round_yellow = find_last_round(board_yellow)
    last_round_red = find_last_round(board_red)
    last_round = np.maximum(last_round_yellow, last_round_red)

    # Keep a copy of the last board
    if last_round_yellow == last_round:
        final_board = board_yellow[last_round]
        board = board_yellow.copy()
        next_turn = 'yellow'
    elif last_round_red == last_round:
        final_board = board_red[last_round]
        board = board_red.copy()
        next_turn = 'red'

    # Keep a Copy of Yellow and Red Tokens
    yellow_idx = np.where(final_board == 1)
    red_idx = np.where(final_board == -1)

    # Determine Winner
    win_flag = False
    blank_board = np.zeros_like(board_yellow[0])
    if next_turn == 'yellow':

        blank_board[yellow_idx] = 1
        blank_board[red_idx] = -1

        win_flag, move_idx = find_winning_moves(blank_board)

        if win_flag is True:
            winner = 'yellow'
        else:
            winner = 'red'
    elif next_turn == 'red':

        blank_board[yellow_idx] = -1
        blank_board[red_idx] = 1

        win_flag, move_idx = find_winning_moves(blank_board)

        if win_flag is True:
            winner = 'red'
        else:
            winner = 'yellow'

    # Store index's of winning and loser player
    if winner == 'yellow':
        win_idx = np.where(board == 1)
        lose_idx = np.where(board == -1)
    elif winner == 'red':
        win_idx = np.where(board == -1)
        lose_idx = np.where(board == 1)

    # Create Winner Board
    winner_board = np.zeros_like(board)
    winner_board[win_idx] = 1
    winner_board[lose_idx] = -1

    # Skip dud matches
    if last_round <= 2:
        total_dud_rounds += 1
        print('Total Dud Matches = {}. Game {} is tiny'.format(total_dud_rounds, game_num))
        continue

    # Iterate over all rounds
    for round in range(last_round+1):
        current_board = winner_board[round]                             # Get the current Board
        move = None                                                     # Set move to none, we can then check that the move is not none as a bug check

        # If we aren't in the last round, determine the move we should make
        if round != last_round:
            next_board = winner_board[round + 1]                        # Get the next board
            board_diff = next_board - current_board                     # Calculate the difference between boards
            board_diff[board_diff == -1] = 0                            # Remove the opponents move for now
            col_sum = np.sum(board_diff, axis=0)                        # Reduce the 2D matrix to 1D Matrix by summing columns. Only 1 column will have a 1
            move = np.where(col_sum == 1)[0]                            # Find the column we should place the token

        # If this is the last Round, determine the winning move
        elif round == last_round:

            any_win_flag, move_idx = find_winning_moves(current_board)

            # If it's the last round and we can't find a winning move, delete game
            if any_win_flag is False:
                total_dud_rounds += 1
                print('Total Dud Matches = {}. No winner in game {}'.format(total_dud_rounds, game_num))
                dud_rounds = round
                blank_board = np.zeros_like(move_history_input[0])
                blank_output = np.zeros_like(move_history_output[0])
                for dud_round in range(dud_rounds):
                    move_history_input[move_count] = blank_board
                    move_history_output[move_count] = blank_output
                    move_count -= 1
                continue

            np.random.shuffle(move_idx)                                 # Shuffle the array
            move = move_idx[0]                                          # Select the shuffled array's first index as the winning move

        # Record History
        move_history_input[move_count] = winner_board[round]            # Set the input data to the current board
        move_history_output[move_count][move] = 1                       # Set the output data to the move made by the winning player
        move_count += 1                                                 # Increment the move count by 1

# Constrain the total number of rounds to the number we have calculated
move_history_input = move_history_input[:move_count-1]
move_history_output = move_history_output[:move_count-1]

# Save Data
np.save(save_folder + 'input_data.npy', move_history_input)
np.save(save_folder + 'output_data.npy', move_history_output)
