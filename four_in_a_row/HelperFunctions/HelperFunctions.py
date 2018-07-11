import numpy as np
import matplotlib.pyplot as plt


def find_last_round(board):
    for round in range(1, board.shape[0]):  # Skip first round (Sum is 0 for the first player's first move)
        board_abs = np.abs(board[round])
        board_sum = np.sum(board_abs)

        if board_sum == 0:
            last_round = round - 1
            break

    return last_round


def plot_board(board):
    plt.figure()
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            if board[5-row][col] == 1:
                plt.scatter(col, row, c='Yellow', s=500, edgecolors='black')

            if board[5-row][col] == -1:
                plt.scatter(col, row, c='Red', s=500, edgecolors='black')
    plt.grid()
    plt.ylim(-1, 6)
    plt.xlim(-1, 7)
    plt.show()


def plot_winner_board(board):
    plt.figure()
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            if board[5-row][col] == 1:
                plt.scatter(col, row, c='Blue', s=500, edgecolors='black')

            if board[5-row][col] == -1:
                plt.scatter(col, row, c='Black', s=500, edgecolors='black')
    plt.grid()
    plt.ylim(-1, 6)
    plt.xlim(-1, 7)
    plt.show()
    plt.close()


def four_in_a_row(board):

    win_flag = False

    # Check Diagonal (P1 = Bottom Left - P4 = Top Right)
    for p1_col in range(4):
        p2_col = p1_col + 1
        p3_col = p2_col + 1
        p4_col = p3_col + 1

        for p1_row in range(3, 6):
            p2_row = p1_row - 1
            p3_row = p2_row - 1
            p4_row = p3_row - 1

            p1 = board[p1_row][p1_col]
            p2 = board[p2_row][p2_col]
            p3 = board[p3_row][p3_col]
            p4 = board[p4_row][p4_col]

            if np.sum([p1, p2, p3, p4]) == 4:
                win_flag = True

    # Check Diagonal (P1 = Top Left - P4 = Bottom Right)
    for p1_col in range(3):
        p2_col = p1_col + 1
        p3_col = p2_col + 1
        p4_col = p3_col + 1

        for p1_row in range(3):
            p2_row = p1_row + 1  # Careful, we swap sign to +
            p3_row = p2_row + 1  # Careful, we swap sign to +
            p4_row = p3_row + 1  # Careful, we swap sign to +

            p1 = board[p1_row][p1_col]
            p2 = board[p2_row][p2_col]
            p3 = board[p3_row][p3_col]
            p4 = board[p4_row][p4_col]

            if np.sum([p1, p2, p3, p4]) == 4:
                win_flag = True

    # Check for row win
    for row in range(board.shape[0]):
        for p1 in range(4):
            p4 = p1 + 3

            section = board[row][p1:p4+1]

            if np.sum(section) == 4:
                win_flag = True

    # Check for column win
    for col in range(board.shape[1]):
        for p1 in range(3):
            p4 = p1 + 3
            section = board[:, col][p1:p4+1]
            if np.sum(section) == 4:
                win_flag = True

    return win_flag


def find_winning_moves(board):

    move_idx = []                                       # List to hold any winning moves
    valid_cols = np.where(board[0] == 0)[0]             # Calculate valid columns we can put token in
    win_flag = False
    any_win_flag = False

    # Iterate over all valid columns
    for col in valid_cols:
        new_board = board.copy()                        # Copy the current board to a new board
        row_options = np.where(board[:, col] == 0)[0]   # For each column, calc which rows are empty
        lowest_row = np.max(row_options)                # The lowest row will be the maximum index row
        new_board[lowest_row][col] = 1                  # Put a token in the new position

        win_flag = four_in_a_row(new_board)             # Check if this play would win the game. Win_flag will be true if so

        # If Win, record column as a winning play
        if win_flag is True:
            any_win_flag = True
            move_idx.append(col)                        # Add the win to the list of possible winning moves

    return any_win_flag, np.array(move_idx)             # Convert list of possible moves to a numpy array