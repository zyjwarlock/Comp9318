import tensorflow as tf
import os
import sys
import numpy as np
import math
import data_process

tf.logging.set_verbosity(tf.logging.INFO)

# file_path = {
#     'train_file' : '',
#     'val_file' : '',
#     'test_file' : '',
#     'vocab_file' : '',
#     'category_file' : '',
#     'output_folder' : 'src/run_text_run'
# }

file_path = {
    "class_0" : 'src/class-0.txt',
    "class_1" : 'src/class-1.txt',
    'output_folder' : 'src/run_text_run'
}


def get_default_params():

    return tf.contrib.training.HParams(#管理模型参数
        num_embedding_size=16, #
        num_timesteps=50,  #how many word every min_batch contains
        num_lstm_nodes=[32, 32], #number of cell in each layer
        num_lstm_layers=2, # number of layer,
        num_fc_nodes=32, #
        batch_size=10, #
        clip_lstm_grads=1.0, # maximum of gradient
        learning_rate=0.001,
        num_word_threshold=10
    )

output_folder = file_path['output_folder']

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

hps = get_default_params()
#vocab = data_process.Vocab(file_path['vocab_file'], hps.num_word_threshold)


def create_model(hps, vocab_size, num_classes, num_timesteps):

    #num_timesteps = hps.num_timesteps
    batch_size = hps.batch_size

    inputs = tf.placeholder(tf.int32, (batch_size, num_timesteps))
    outputs = tf.placeholder(tf.int32, (batch_size, ))
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob') #dropout = 1 - keep_prob

    global_step = tf.Variable(tf.zeros([], tf.int64), name = 'global_step', trainable=False) #保存训练到哪一步

    embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)
    with tf.variable_scope('embedding', initializer=embedding_initializer):
        embeddings = tf.get_variable(
            'embedding',
            [vocab_size, hps.num_embedding_size],
            tf.float32
        )

        embed_inputs = tf.nn.embedding_lookup(embeddings, inputs)

    scale = 1.0 / math.sqrt(hps.num_embedding_size + hps.num_lstm_nodes[-1]) / 3.0
    lstm_init = tf.random_uniform_initializer(-scale, scale)

    with tf.variable_scope('lstm_nn', initializer = lstm_init):
        cells = []
        for i in range(hps.num_lstm_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(
                hps.num_lstm_nodes[i],
                state_is_tuple = True
            )

            cell = tf.contrib.rnn.DropoutWrapper(
                cell,
                output_keep_prob = keep_prob)

            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)
        #rnn_outputs: [batch_size, num_timesteps, lstm_outputs[-1]]
        initial_state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, _ = tf.nn.dynamic_rnn(cell, embed_inputs, initial_state = initial_state)
        last = rnn_outputs[:, -1, :]

    fc_init = tf.uniform_unit_scaling_initializer(factor = 1.0)

    with tf.variable_scope('fc', initializer = fc_init):
        fc1 = tf.layers.dense(last,
                              hps.num_fc_nodes,
                              activation = tf.nn.relu,
                              name = 'fc1')
        fc1_dropout = tf.contrib.layers.dropout(fc1, keep_prob)
        logits = tf.layers.dense(fc1_dropout,
                                 num_classes,
                                 name = 'fc2')
    with tf.name_scope('metrices'):

        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=outputs)
        loss = tf.reduce_mean(softmax_loss)
        y_pred = tf.argmax(tf.nn.softmax(logits), 1, output_type=tf.int32)
        correct_pred = tf.equal(outputs, y_pred)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.name_scope('train_op'):
        tvars = tf.trainable_variables()
        for var in tvars:
            tf.logging.info('variable name: %s' % (var.name))
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss, tvars), hps.clip_lstm_grads)
        optimizer = tf.train.AdamOptimizer(hps.learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    return ((inputs, outputs, keep_prob), (loss, accuracy), (train_op, global_step))

train_dataset = data_process.Data_process(file_path['class_0'], file_path['class_1'])
placeholders, metrics, others = create_model(hps, train_dataset.vocab_size(), train_dataset.num_classes(), train_dataset.num_timestep())
inputs, outputs, keep_prob = placeholders
loss, accuracy = metrics
train_op, globals_step = others

#train model

init_op = tf.global_variables_initializer()
train_keep_prob_value = 0.8
test_keep_prob_value = 1.0

num_train_steps = 10000

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(num_train_steps):
        batch_inputs, batch_labels = train_dataset.next_batch(hps.batch_size)
        outputs_val = sess.run([loss, accuracy, train_op, globals_step],
                           feed_dict={
                               inputs: batch_inputs,
                               outputs: batch_labels,
                               keep_prob: train_keep_prob_value,
                           })
        loss_val, accuracy_val, _, globals_step_val = outputs_val
        if globals_step_val % 20 ==0:
            tf.logging.info("step: %5d, loss: %3.3f, accuary: %3.3f" %
                            (globals_step_val, loss_val, accuracy_val))

 
