'''
Created on Jul 2, 2017

@author: SamRagusa
'''

import tensorflow as tf
import numpy as np
import threading
import time

import ann_creation_helper as ann_h
from chestimator import ChEstimator
tf.logging.set_verbosity(tf.logging.INFO)



def main(using_to_serve):
    """
    Set up the data pipelines, create the computational graph, train the model, and evaluate the results.
    """
    SAVE_MODEL_DIR = "/srv/tmp/current/crazy_network_shape_idea_12"#"/srv/tmp/current/smaller_eval_test_5"
    TRAINING_FILENAMES = ["/srv/databases/chess_engine/full_3/shuffled_training_set.txt"]
    VALIDATION_FILENAMES = ["/srv/databases/chess_engine/full_5/scoring_validation_set.tfrecords"]
    TRAIN_OP_SUMMARIES = ["gradient_norm", "gradients"]
    NUM_OUTPUTS = 1
    DENSE_SHAPE = [500,250,100]#[500,300,100]#[500,400,300,100]#[500,350,250]
    OPTIMIZER = 'Adam'
    TRAINING_MIN_AFTER_DEQUEUE = 1000000
    TRAINING_BATCH_SIZE = 500#1000#500
    VALIDATION_BATCH_SIZE = 2000
    NUM_TRAINING_EPOCHS = 500
    LOG_ITERATION_INTERVAL =2000
    OLD_MOVE_SCALAR_MULT = 1.00
    LEARNING_RATE = .00001#.00001#.000002


    INCEPTION_MODULES = [
        [
            [[25, 2], [50, 2]],
            [[50, 3]]],
        [
            [[50, 1]],
            [[15, 1], [30, 1, 6]],
            [[15, 1], [30, 6, 1]]],
        lambda tensor_list: tf.concat([
            tensor_list[0],
            tf.concat([tensor_list[1] for _ in range(6)], 2),
            tf.concat([tensor_list[2] for _ in range(6)], 1)], 3),
        [
            [[50,1], [750,6]]
        ],
        lambda x : tf.reshape(x, [-1,750]),
    ]






    BATCHES_IN_TRAINING_EPOCH = 758545383  // TRAINING_BATCH_SIZE
    #reduce(
        # lambda x, y: x + y,
        # [ann_h.line_counter(filename) for filename in TRAINING_FILENAMES]) // (TRAINING_BATCH_SIZE * 10)
    BATCHES_IN_VALIDATION_EPOCH = 1000000// VALIDATION_BATCH_SIZE
    #84289074 // VALIDATION_BATCH_SIZE
    #ann_h.line_counter(VALIDATION_FILENAMES[0]) // VALIDATION_BATCH_SIZE


    learning_decay_function = lambda gs  : tf.train.exponential_decay(LEARNING_RATE,
                                                                      global_step=gs,
                                                                      decay_steps=BATCHES_IN_TRAINING_EPOCH//4,
                                                                      decay_rate=0.93,
                                                                      staircase=True)


    print(BATCHES_IN_TRAINING_EPOCH)
    print(BATCHES_IN_VALIDATION_EPOCH)


    # Create the Estimator
    the_estimator = ChEstimator(
        model_fn=ann_h.cnn_model_fn,
        model_dir=SAVE_MODEL_DIR,
        config=tf.estimator.RunConfig().replace(
            save_checkpoints_steps=LOG_ITERATION_INTERVAL,
            save_summary_steps=LOG_ITERATION_INTERVAL),
            # session_config=tf.ConfigProto(log_device_placement=True)),
        params={
            "dense_shape": DENSE_SHAPE,
            "optimizer": OPTIMIZER,
            "num_outputs": NUM_OUTPUTS,
            "log_interval": LOG_ITERATION_INTERVAL,
            "model_dir": SAVE_MODEL_DIR,
            "inception_modules" : INCEPTION_MODULES,
            "learning_rate": LEARNING_RATE,
            "train_summaries": TRAIN_OP_SUMMARIES,
            "learning_decay_function" : learning_decay_function,
            "inc_old_move_scalar" : OLD_MOVE_SCALAR_MULT,
            })



    if using_to_serve[0]:
        return the_estimator.all_white_create_predictor( "scores")

    validation_hook = ann_h.ValidationRunHook(
        step_increment=BATCHES_IN_TRAINING_EPOCH//(5*10),
        estimator=the_estimator,
        input_fn_creator=lambda: ann_h.one_hot_create_tf_records_input_data_fn(VALIDATION_FILENAMES,VALIDATION_BATCH_SIZE),
        temp_num_steps_in_epoch=500)#BATCHES_IN_VALIDATION_EPOCH)

    the_estimator.train(
        input_fn=lambda: ann_h.input_data_fn(
            TRAINING_FILENAMES,
            TRAINING_BATCH_SIZE,
            NUM_TRAINING_EPOCHS,
            TRAINING_MIN_AFTER_DEQUEUE,
            True),
        hooks=[validation_hook],
        # max_steps=1,
    )



    # # Export the model for serving for whites turn
    # the_estimator.export_savedmodel(
    #     SAVE_MODEL_DIR + "/whites_turn",
    #     serving_input_receiver_fn=ann_h.serving_input_reciever_fn_creater(True)
    # )
    #
    # # Export the model for serving for blacks turn
    # the_estimator.export_savedmodel(
    #     SAVE_MODEL_DIR + "/blacks_turn",
    #     serving_input_receiver_fn=ann_h.serving_input_reciever_fn_creater(False)
    # )




if __name__ == "__main__":
    tf.app.run(argv=(None, False,))