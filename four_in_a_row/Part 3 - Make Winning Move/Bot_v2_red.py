#!/usr/bin/env python3
import sys
import numpy as np
import os

# Player will choose a winning move if possible

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

class Settings(object):
    def __init__(self):
        self.timebank = None
        self.time_per_move = None
        self.player_names = None
        self.your_bot = None
        self.your_botid = None
        self.field_width = None
        self.field_height = None


class Field(object):
    def __init__(self):
        self.field_state = None

    def update_field(self, celltypes, settings):
        self.field_state = [[] for _ in range(settings.field_height)]
        n_cols = settings.field_width
        for idx, cell in enumerate(celltypes):
            row_idx = idx // n_cols
            self.field_state[row_idx].append(cell)

class State(object):
    def __init__(self):
        self.settings = Settings()
        self.field = Field()
        self.round = 0

        folder = './Saved_Games_Red/'
        num_files = os.listdir(folder)
        self.name = folder + str(len(num_files)+1) + '.npy'
        self.first = None


def parse_communication(text):
    """ Return the first word of the communication - that's the command """
    return text.strip().split()[0]


def settings(text, state):
    """ Handle communication intended to update game settings """
    tokens = text.strip().split()[1:]  # Ignore token 0, it's the string "settings".
    cmd = tokens[0]
    if cmd in ('timebank', 'time_per_move', 'your_botid', 'field_height', 'field_width'):
        # Handle setting integer settings.
        setattr(state.settings, cmd, int(tokens[1]))
    elif cmd in ('your_bot',):
        # Handle setting string settings.
        setattr(state.settings, cmd, tokens[1])
    elif cmd in ('player_names',):
        # Handle setting lists of strings.
        setattr(state.settings, cmd, tokens[1:])
    else:
        raise NotImplementedError('Settings command "{}" not recognized'.format(text))


def update(text, state):
    """ Handle communication intended to update the game """
    tokens = text.strip().split()[2:] # Ignore tokens 0 and 1, those are "update" and "game" respectively.
    cmd = tokens[0]
    if cmd in ('round',):
        # Handle setting integer settings.
        setattr(state.settings, 'round', int(tokens[1]))
    if cmd in ('field',):
        # Handle setting the game board.
        celltypes = tokens[1].split(',')
        state.field.update_field(celltypes, state.settings)


def action(text, state):
    """ Handle communication intended to prompt the bot to take an action """
    tokens = text.strip().split()[1:] # Ignore token 0, it's the string "action".
    cmd = tokens[0]
    if cmd in ('move',):
        move = make_move(state)
        state.round += 1
        return move
    else:
        raise NotImplementedError('Action command "{}" not recognized'.format(text))

def make_move(state):

    # Convert field state into a numpy array
    field_state = np.array(state.field.field_state)

    # Recording Data
    if state.round == 0:
        game_history = np.zeros((30, 6, 7))
    else:
        game_history = np.load(state.name)

    ## Record Game History. 1 = Yellow Tokens, -1 = Red Tokens, 0 = Empty Spot
    # Iterate over all rows
    for row in range(6):
        yellow_idx = np.where(field_state[row] == '1')[0]
        red_idx = np.where(field_state[row] == '0')[0]

        game_history[state.round][row][yellow_idx] = 1
        game_history[state.round][row][red_idx] = -1

    # Choose a random move out of all valid columns
    valid_idx = np.where(game_history[state.round][0] == 0)[0]
    np.random.shuffle(valid_idx)
    move = valid_idx[0]

    # Determine who moved first
    if state.round == 1:
        token_idx = np.where(field_state[5] != '.')[0]
        r1_tokens = np.array(field_state[5][token_idx]).astype(float)

        if np.sum(r1_tokens) == 1 and r1_tokens.shape[0] == 2:    # Red First, Red See's
            state.first = 'Red'
        elif np.sum(r1_tokens) == 1 and r1_tokens.shape[0] == 3:  # Red First, Yellow See's
            state.first = 'Red'

        elif np.sum(r1_tokens) == 1 and r1_tokens.shape[0] == 1:  # Yellow First, Red See's
            state.first = 'Yellow'
        elif np.sum(r1_tokens) == 2 and r1_tokens.shape[0] == 3:  # Yellow First, Yellow See's
            state.first = 'Yellow'

    ## Before we move, let's check if there is a move we can make to win!

    # 1. Determine who's turn it is
    # If sum = -1, More Yellow Tokens -> Red's turn
    if np.sum(game_history[state.round]) == -1:
        me_idx = np.where(game_history[state.round] == 1)     # These might be wrong
        enemy_idx = np.where(game_history[state.round] == -1) # These might be wrong

    # If sum = 0 -> Even # Tokens & Yellow moved first --> Mean's it's Yellow's turn
    elif np.sum(game_history[state.round]) == 0 and state.first == 'Yellow':
        me_idx = np.where(game_history[state.round] == 1)

        enemy_idx = np.where(game_history[state.round] == -1)
    # If sum = 0, Even # Tokens & Red moved first
    elif np.sum(game_history[state.round]) == 0:
        me_idx = np.where(game_history[state.round] == -1)
        enemy_idx = np.where(game_history[state.round] == 1)


    # 2. Organise the board so that 1 = Me, -1 = Enemy.
    me_board = np.zeros_like(game_history[state.round])
    me_board[me_idx] = 1
    me_board[enemy_idx] = -1


    # 3. Check if any columns are winning columns:
    for col_idx in valid_idx:
        row_idx = np.where(me_board[:, col_idx] == 0)[0]
        row_idx = np.max(row_idx)
        me_board[row_idx][col_idx] = 1
        win_flag = four_in_a_row(me_board)
        if win_flag is True:
            move = col_idx
            break

    np.save(state.name, game_history)

    return 'place_disc {}'.format(move)

def main():
    command_lookup = { 'settings': settings, 'update': update, 'action': action }
    state = State()
    for input_msg in sys.stdin:
        cmd_type = parse_communication(input_msg)
        command = command_lookup[cmd_type]

        # Call the correct command.
        res = command(input_msg, state)

        # Assume if the command generates a string as output, that we need
        # to "respond" by printing it to stdout.
        if isinstance(res, str):
            print(res)
            sys.stdout.flush()



if __name__ == '__main__':
    main()
