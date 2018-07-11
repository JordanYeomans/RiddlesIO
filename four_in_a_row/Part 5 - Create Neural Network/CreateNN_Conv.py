import numpy as np
import matplotlib.pyplot as plt
import time
from HelperFunctions.HelperFunctions import plot_winner_board

import tensorflow as tf

def split_data(move_history_input, move_history_output, val_split, eval_split):

    num_val = int(move_history_input.shape[0] * val_split)
    num_eval = int(move_history_input.shape[0] * eval_split)
    num_train = move_history_input.shape[0] - num_val - num_eval

    all_idx = np.arange(0, move_history_input.shape[0], 1)

    np.random.shuffle(all_idx)

    train_idx = all_idx[:num_train]
    val_idx = all_idx[num_train: num_train + num_val]
    eval_idx = all_idx[num_train + num_val:]

    train_input_data, train_output_data = move_history_input[train_idx], move_history_output[train_idx]
    val_input_data, val_output_data = move_history_input[val_idx], move_history_output[val_idx]
    eval_input_data, eval_output_data = move_history_input[eval_idx], move_history_output[eval_idx]

    print('Training Samples = {}, Validation Samples = {}, Evaluation Samples = {}'.format(train_input_data.shape[0], val_input_data.shape[0], eval_input_data.shape[0]))
    return train_input_data, train_output_data, val_input_data, val_output_data, eval_input_data, eval_output_data

def split_into_batches(train_in_data, train_out_data, batch_size):

    num_batches = int(np.floor(train_in_data.shape[0]/batch_size))

    train_in_batches = np.zeros((num_batches, batch_size, train_in_data.shape[1], train_in_data.shape[2]))
    train_out_batches = np.zeros((num_batches, batch_size, train_out_data.shape[1]))

    samples = np.arange(0, num_batches * batch_size, step=1)
    np.random.shuffle(samples)

    for i in range(train_in_batches.shape[0]):
        start_num = i * batch_size
        end_num = start_num + batch_size

        batch_samples = samples[start_num:end_num]

        train_in_batches[i] = train_in_data[batch_samples]
        train_out_batches[i] = train_out_data[batch_samples]

    return train_in_batches, train_out_batches

def add_last_dim(input_data):
    input_data = input_data.reshape([input_data.shape[0], input_data.shape[1], input_data.shape[2], 1])
    return input_data

# Folder Paths
data_load_folder = 'C:/Users/Jordan Yeomans/Documents/GitHub/RiddlesIO/four_in_a_row/Data/Processed_Data/Random_vs_Random/'

# NN Parameters
lr = 0.001
epochs = 1000
val_split = 0.1
eval_split = 0.05
batch_size = 1024

### Code Starts Here ###
# Load Data
move_history_input = np.load(data_load_folder + 'input_data.npy')
move_history_output = np.load(data_load_folder + 'output_data.npy')

# Check Number Of Samples
print('Input Data Shape = {}:'.format(move_history_input.shape))
print('Output Data Shape = {}:'.format(move_history_output.shape))

# Reshape output to shape (7, )
move_history_output = move_history_output.reshape(move_history_output.shape[0], move_history_output.shape[1])

# Split data into training, validation and evaluation
train_in_data, train_out_data, val_in_data, val_out_data, eval_in_data, eval_out_data = split_data(move_history_input, move_history_output, val_split, eval_split)

# Look at an example Input/Output


# Create Placeholders
x = tf.placeholder(tf.float32, shape=[None, 6, 7,1], name='input_placeholder')
y = tf.placeholder(tf.float32, shape=[None, 7], name='output_placeholder')

# Create Network
initial = tf.initializers.truncated_normal(stddev=0.05)

# nn = tf.layers.flatten(x, name='input')
# nn = tf.layers.dense(nn, 256, activation=tf.nn.relu, kernel_initializer=initial)
# nn = tf.layers.dropout(nn, 0.5)
# nn = tf.layers.dense(nn, 256, activation=tf.nn.relu, kernel_initializer=initial)
# nn = tf.layers.dropout(nn, 0.5)
# nn = tf.layers.dense(nn, 256, activation=tf.nn.relu, kernel_initializer=initial)
# nn = tf.layers.dropout(nn, 0.5)
# last_layer = tf.layers.dense(nn, 7, name='output')

#nn = tf.layers.flatten(x, name='input')

nn = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=initial)
nn = tf.layers.dropout(nn, 0.5)
nn = tf.layers.conv2d(nn, filters=32, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=initial)
nn = tf.layers.dropout(nn, 0.5)
nn = tf.layers.conv2d(nn, filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=initial)

nn = tf.layers.flatten(nn)
nn = tf.layers.dense(nn, 128, activation=tf.nn.relu, kernel_initializer=initial)
nn = tf.layers.dropout(nn, 0.5)
nn = tf.layers.dense(nn, 64, activation=tf.nn.relu, kernel_initializer=initial)
nn = tf.layers.dropout(nn, 0.5)
nn = tf.layers.dense(nn, 32, activation=tf.nn.relu, kernel_initializer=initial)
nn = tf.layers.dropout(nn, 0.5)

last_layer = tf.layers.dense(nn, 7, name='output')

# Define Loss Function
loss_function = tf.nn.softmax_cross_entropy_with_logits_v2(logits=last_layer, labels=y)
loss = tf.reduce_mean(loss_function)

# Create Optimiser
learning_step = tf.train.AdamOptimizer(lr)
optimiser = learning_step.minimize(loss)

# Create Accuracy Evaluation
correct = tf.equal(tf.argmax(last_layer, axis=1), tf.argmax(y, axis=1))
acc = tf.reduce_mean(tf.cast(correct, 'float'))

val_in_data = add_last_dim(val_in_data)
eval_in_data = add_last_dim(eval_in_data)
train_check_in_data = add_last_dim(train_in_data)

# Train Network
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        # Split training data into batches
        train_in_batches, train_out_batches = split_into_batches(train_in_data, train_out_data, batch_size)

        for batch in range(train_in_batches.shape[0]):

            input_data = add_last_dim(train_in_batches[batch])

            _, batch_cost = sess.run([optimiser, loss], feed_dict={x: input_data, y: train_out_batches[batch]})

        epoch_train_acc = sess.run(acc, feed_dict={x: train_check_in_data, y: train_out_data})
        epoch_val_acc = sess.run(acc, feed_dict={x: val_in_data, y: val_out_data})

        if epoch % 10 == 0:
            print('Single Batch Cost = {:.3f}, Train Acc = {:.3f}%, Val Acc = {:.3f}%'.format(batch_cost, epoch_train_acc * 100, epoch_val_acc * 100))

    # After Training, Check Evaluation Accuracy
    eval_acc = sess.run(acc, feed_dict={x: eval_in_data, y: eval_out_data})
    print('Evaluation Acc = {}', format(eval_acc))