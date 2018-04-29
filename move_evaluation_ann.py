import tensorflow as tf
import numpy as np
from scipy import stats

import ann_creation_helper as ann_h
from functools import reduce
from chestimator import MoveChEstimator
tf.logging.set_verbosity(tf.logging.INFO)




def main(using_to_serve):
    """
    Set up the data pipelines, create the computational graph, train the model, and evaluate the results.
    """
    SAVE_MODEL_DIR = "/srv/tmp/very_current/very_new_data_move_ordering_10"#"/srv/tmp/current/pre_trained_conv_layers_15"
    TRAINING_FILENAME_PATTERN = "/srv/databases/chess_engine/one_rand_per_board_data/move_scoring_training_set_*.tfrecords"#"/srv/databases/chess_engine/new_move_gen_new_mapping/testing_stuff_split_*.tfrecords"
    VALIDATION_FILENAME_PATTERN = "/srv/databases/chess_engine/one_rand_per_board_data/move_scoring_validation_set_*.tfrecords"#"/srv/databases/chess_engine/new_move_gen_new_mapping/testing_stuff_split_*.tfrecords"#["/srv/databases/chess_engine/move_gen/full_training_data_part_9.tfrecords"]
    TESTING_FILENAME_PATTERN = "/srv/databases/chess_engine/one_rand_per_board_data/move_scoring_testing_set_*.tfrecords"
    TRAIN_OP_SUMMARIES = ["gradient_norm", "gradients"]
    NUM_INPUT_FILTERS = 15
    NUM_OUTPUTS = 1792
    DENSE_SHAPE =  [1024,1024,1024,1024,1024,800,800]#[1024,1024,1024,1024,800,800]#[2048,2048,1500,1024,1024,1024]#[2048,2048,1500,1024,1024,1024,800]#[2048,2048,1024,1024]#500,500,500,800]#[1000,1000,1000]#[800,800,800]
    OPTIMIZER = 'Adam'
    TRAINING_BATCH_SIZE = 200#500
    VALIDATION_BATCH_SIZE = 5000
    # TESTING_BATCH_SIZE = 10000
    LOG_ITERATION_INTERVAL = 2000
    LEARNING_RATE = .00001#.000001#.000002
    MAKE_CNN_MODULES_TRAINABLE = True
    # CHECKPOINT_DIR_WITH_CONV_LAYERS = "/srv/tmp/current/small_cnn_2"
    init_conv_layers_fn = None


    INCEPTION_MODULES = [
        [
            [[25, 2], [50, 2]],
            [[50, 3]]],
        [
            [[50, 1]],
            [[15, 1], [30, 1, 6]],
            [[15, 1], [30, 6, 1]]]]


    BATCHES_IN_TRAINING_EPOCH = 3870000 // (TRAINING_BATCH_SIZE)
    BATCHES_IN_VALIDATION_EPOCH = 430000// VALIDATION_BATCH_SIZE


    learning_decay_function = lambda gs  : tf.train.exponential_decay(LEARNING_RATE,
                                                                      global_step=gs,
                                                                      decay_steps=2*BATCHES_IN_TRAINING_EPOCH,
                                                                      decay_rate=0.96,
                                                                      staircase=True)


    # scopes_to_restore = ["inception_module_1_path_1", "inception_module_1_path_2","inception_module_2_path_3", "inception_module_2_path_2", "inception_module_2_path_3", "inception_module_4_path_1"]
    # dict_things_to_restore = {name + "/": name + "/" for name in scopes_to_restore}
    #
    # init_conv_layers_fn = lambda: tf.train.init_from_checkpoint(CHECKPOINT_DIR_WITH_CONV_LAYERS, dict_things_to_restore)


    # Create the Estimator
    the_estimator = MoveChEstimator(
        model_fn=ann_h.move_gen_cnn_model_fn,
        model_dir=SAVE_MODEL_DIR,
        config=tf.estimator.RunConfig().replace(
            save_checkpoints_steps=LOG_ITERATION_INTERVAL,
            save_summary_steps=LOG_ITERATION_INTERVAL,
            session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.25))),
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
            "trainable_cnn_modules" : MAKE_CNN_MODULES_TRAINABLE,
            "conv_init_fn" : init_conv_layers_fn,
            "behold" : None
        })


    if using_to_serve[0]:
        return the_estimator.create_move_predictor("move_values")



    validation_hook = ann_h.ValidationRunHook(
        step_increment=BATCHES_IN_TRAINING_EPOCH,
        estimator=the_estimator,
        input_fn_creator=lambda: ann_h.move_gen_one_hot_create_tf_records_input_data_fn(VALIDATION_FILENAME_PATTERN,VALIDATION_BATCH_SIZE,include_unoccupied=NUM_INPUT_FILTERS==16,repeat=False,shuffle=False),
        temp_num_steps_in_epoch=BATCHES_IN_VALIDATION_EPOCH,
        recall_input_fn_creator_after_evaluate=True)



    the_estimator.train(
        input_fn=ann_h.move_gen_one_hot_create_tf_records_input_data_fn(
            TRAINING_FILENAME_PATTERN,
            TRAINING_BATCH_SIZE,
            include_unoccupied=NUM_INPUT_FILTERS==16),
        hooks=[validation_hook],
        # max_steps=1,
    )





    # the_estimator.export_savedmodel(
    #     SAVE_MODEL_DIR + "/whites_turn",
    #     serving_input_receiver_fn=ann_h.serving_input_reciever_legal_moves_fn(True),
    # )
    #
    # the_estimator.export_savedmodel(
    #     SAVE_MODEL_DIR + "/blacks_turn",
    #     serving_input_receiver_fn=ann_h.serving_input_reciever_legal_moves_fn(False),
    # )





if __name__ == "__main__":
    tf.app.run(argv=[False])