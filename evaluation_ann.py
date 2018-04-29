'''
Created on Jul 2, 2017

@author: SamRagusa
'''

import tensorflow as tf

import ann_creation_helper as ann_h
from chestimator import ChEstimator

tf.logging.set_verbosity(tf.logging.INFO)



def main(using_to_serve):
    """
    Set up the data pipelines, create the computational graph, train the model, and evaluate the results.
    """
    SAVE_MODEL_DIR = "/srv/tmp/very_current/pre_commit_test"#"/srv/tmp/very_current/mse_with_cross_entropy_4"#two_percent_equality_diff_1"#"/srv/tmp/very_current/even_crazier_shit_5"#"#"/srv/tmp/very_current/blahblah_2"#"/srv/tmp/very_current/super_weird_loss_5"#"/srv/tmp/very_current/crazy_older_graph_12"#"/srv/anns_to_maybe_use/no_conv_for_real_test_5.2"#"/srv/tmp/current/no_conv_for_real_test_5"#"/srv/tmp/current/no_empty_filter_1"#"/srv/tmp/current/some_name_4"#"/srv/tmp/current/newer_conv_data_7"#"/srv/anns_to_maybe_use/newer_data_4"##"/srv/tmp/current/beholder_test_3"##"/srv/tmp/current/no_conv3.0"#"/srv/anns_to_maybe_use/no_conv_2.15_older"#"/srv/tmp/current/no_conv_2.15"#"/srv/tmp/current/conv_high_equality2"#"/srv/tmp/current/no_conv_2.10"#
    TRAINING_FILENAME_PATTERN = "/srv/databases/chess_engine/one_rand_per_board_data/scoring_training_set_*.tfrecords"#"/srv/databases/chess_engine/full_8/scoring_training_set_*.tfrecords"#["/srv/databases/chess_engine/full_7/scoring_training_set_"+str(j)+".tfrecords" for j in range(1,7)]#["/srv/databases/chess_engine/full_3/shuffled_training_set.txt"]
    VALIDATION_FILENAME_PATTERN = "/srv/databases/chess_engine/one_rand_per_board_data/scoring_validation_set_*.tfrecords"#"/srv/databases/chess_engine/full_8/scoring_validation_set_*.tfrecords"#["/srv/databases/chess_engine/full_5/scoring_validation_set.txt"]
    TRAIN_OP_SUMMARIES = ["gradient_norm", "gradients"]
    NUM_INPUT_FILTERS = 15
    NUM_OUTPUTS = 1
    DENSE_SHAPE = [1024,800,650,512]#[1024,1024,1024,512,256,128]#[500,300,100]#[1024,1024,1024,512,256,128]#[1024,1024,1024]###[1024,1024,1024,512,256,128]#[1024,1024,1024,800,512,512]#[1024,1024,1024,512,256,128]#[500,300,100]
    OPTIMIZER = 'Adam'
    TRAINING_SHUFFLE_BUFFER_SIZE = 1000000
    TRAINING_BATCH_SIZE =250#500#1000#500
    VALIDATION_BATCH_SIZE = 20000
    # NUM_TRAINING_EPOCHS = 250
    LOG_ITERATION_INTERVAL =2000
    # OLD_MOVE_SCALAR_MULT = 1.00
    LEARNING_RATE = 1e-6#.00001#.000001#.0000001#.00001#.000002


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






    BATCHES_IN_TRAINING_EPOCH = 14066104 // TRAINING_BATCH_SIZE
    BATCHES_IN_VALIDATION_EPOCH = 2009797// VALIDATION_BATCH_SIZE


    learning_decay_function = lambda gs  : tf.train.exponential_decay(LEARNING_RATE,
                                                                      global_step=gs,
                                                                      decay_steps=BATCHES_IN_TRAINING_EPOCH*5,
                                                                      decay_rate=0.96,
                                                                      staircase=True)



    # Create the Estimator
    the_estimator = ChEstimator(
        model_fn=ann_h.board_eval_model_fn,
        model_dir=SAVE_MODEL_DIR,
        config=tf.estimator.RunConfig().replace(
            save_checkpoints_steps=LOG_ITERATION_INTERVAL,
            save_summary_steps=LOG_ITERATION_INTERVAL,
            session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.6))),#log_device_placement=True)),
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
            "num_input_filters" : NUM_INPUT_FILTERS,
            # "inc_old_move_scalar" : OLD_MOVE_SCALAR_MULT,
            })



    if using_to_serve[0]:
        return the_estimator.create_predictor("scores")

    validation_hook = ann_h.ValidationRunHook(
        step_increment=BATCHES_IN_TRAINING_EPOCH//2,
        estimator=the_estimator,
        input_fn_creator=lambda: ann_h.one_hot_create_tf_records_input_data_fn(VALIDATION_FILENAME_PATTERN, VALIDATION_BATCH_SIZE, include_unoccupied=NUM_INPUT_FILTERS == 16),
        temp_num_steps_in_epoch=BATCHES_IN_VALIDATION_EPOCH)



    the_estimator.train(
        input_fn=ann_h.one_hot_create_tf_records_input_data_fn(TRAINING_FILENAME_PATTERN, TRAINING_BATCH_SIZE, shuffle_buffer_size=TRAINING_SHUFFLE_BUFFER_SIZE, include_unoccupied=NUM_INPUT_FILTERS == 16),
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
    tf.app.run(argv=[False])