'''
AI Learns to Play Connect 4
Part 6 - Implement A1 V1
Jordan Yeomans - 2018
'''

#!/usr/bin/env python3
import sys
import numpy as np
import os
import time
import tensorflow as tf

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

        folder = 'C:/Users/Jordan Yeomans/Documents/GitHub/RiddlesIO/four_in_a_row/Data/Raw_Data/AI-v1_vs_Random/Red_AI-v1/'
        num_files = os.listdir(folder)
        self.name = folder + str(len(num_files)+1) + '.npy'
        self.first = None

        self.model_folder = 'C:/Users/Jordan Yeomans/Documents/GitHub/RiddlesIO/four_in_a_row/NeuralNetworks/AI_Bot_Version_1/'

        self.x = tf.placeholder(tf.float32, shape=[None, 6, 7], name='input_placeholder')
        self.y = tf.placeholder(tf.float32, shape=[None, 7], name='output_placeholder')

        nn = tf.layers.flatten(self.x, name='input')
        nn = tf.layers.dense(nn, 256, activation=tf.nn.relu)
        nn = tf.layers.dense(nn, 256, activation=tf.nn.relu)
        nn = tf.layers.dense(nn, 256, activation=tf.nn.relu)
        self.last_layer = tf.layers.dense(nn, 7, name='output')

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

    field_state = np.array(state.field.field_state)

    # Recording Data
    if state.round == 0:
        current_board = np.zeros((30, 6, 7))
    else:
        current_board = np.load(state.name)

    # Get Most Recent Board
    for row in range(6):
        yellow_idx = np.where(field_state[row] == '1')[0]
        red_idx = np.where(field_state[row] == '0')[0]

        current_board[state.round][row][yellow_idx] = 1
        current_board[state.round][row][red_idx] = -1

    # Make Move from NN
    saver = tf.train.Saver()

    input_data = current_board[state.round]
    input_data = input_data.reshape([1, input_data.shape[0], input_data.shape[1]])

    with tf.Session() as sess:

        saver.restore(sess, state.model_folder)

        output = sess.run(state.last_layer, feed_dict={state.x: input_data})
        move = np.argmax(output)

    np.save(state.name, current_board)

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
