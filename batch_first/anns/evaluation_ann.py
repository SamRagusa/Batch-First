'''
Created on Jul 2, 2017

@author: SamRagusa
'''

import tensorflow as tf



import ann_creation_helper as ann_h


tf.logging.set_verbosity(tf.logging.INFO)





def main(using_to_serve):
    """
    Set up the data pipelines, create the computational graph, train the model, and evaluate the results.
    """
    SAVE_MODEL_DIR = "/srv/tmp/encoder_evaluation/normal_next_try_3"#"/srv/tmp/encoder_evaluation/conv_train_wide_and_deep_4"
    TRAINING_FILENAME_PATTERN = "/srv/databases/chess_engine/one_rand_per_board_data/scoring_training_set_*.tfrecords"
    VALIDATION_FILENAME_PATTERN = "/srv/databases/chess_engine/one_rand_per_board_data/scoring_validation_set_*.tfrecords"
    TRAIN_OP_SUMMARIES = ["gradient_norm", "gradients"]
    NUM_INPUT_FILTERS = 15
    NUM_OUTPUTS = 1
    DENSE_SHAPE = [800,600,400,200,100]
    OPTIMIZER = 'Adam'
    TRAINING_SHUFFLE_BUFFER_SIZE = 5000000
    TRAINING_BATCH_SIZE = 64 #128
    VALIDATION_BATCH_SIZE = 10000
    # NUM_TRAINING_EPOCHS = 250
    LOG_ITERATION_INTERVAL =10000
    LEARNING_RATE = 1e-5
    init_conv_layers_fn = None
    CHECKPOINT_DIR_WITH_CONV_LAYERS = "/srv/tmp/encoder_helper/with_moves_6.4"
    KERNEL_INITIALIZER = lambda : tf.contrib.layers.variance_scaling_initializer()#factor=.5)
    MAKE_CNN_MODULES_TRAINABLE = True

    INCEPTION_MODULES = [
        [
            [[20, 3],  # 720
             [30, 3],  # 480
             [300, 4]]],
        lambda tensor: tf.reshape(tensor, [-1,300]),
    ]



    BATCHES_IN_TRAINING_EPOCH = (7*2009392) // TRAINING_BATCH_SIZE
    BATCHES_IN_VALIDATION_EPOCH = (2009392) // VALIDATION_BATCH_SIZE


    learning_decay_function = lambda gs  : tf.train.exponential_decay(LEARNING_RATE,
                                                                      global_step=gs,
                                                                      decay_steps=BATCHES_IN_TRAINING_EPOCH,#*2,
                                                                      decay_rate=0.96,
                                                                      staircase=True)


    # scopes_to_restore = ["inception_module_1_path_1", ]#"inception_module_1_path_2","inception_module_2_path_1", "inception_module_2_path_2", "inception_module_2_path_3", "inception_module_4_path_1/layer_1","inception_module_4_path_1/batch_normalization"]
    # dict_things_to_restore = {name + "/": name + "/" for name in scopes_to_restore}
    #
    # init_conv_layers_fn = lambda: tf.train.init_from_checkpoint(CHECKPOINT_DIR_WITH_CONV_LAYERS, dict_things_to_restore)



    # Create the Estimator
    the_estimator = tf.estimator.Estimator(
        model_fn=ann_h.board_eval_model_fn,
        model_dir=SAVE_MODEL_DIR,
        config=tf.estimator.RunConfig().replace(
            save_checkpoints_steps=LOG_ITERATION_INTERVAL,
            save_summary_steps=LOG_ITERATION_INTERVAL,#),
            session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.3))),#log_device_placement=True)),
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
            "conv_init_fn": init_conv_layers_fn,
            "kernel_initializer" : KERNEL_INITIALIZER,
            "trainable_cnn_modules" : MAKE_CNN_MODULES_TRAINABLE,
            })



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


    # Save the model for serving
    the_estimator.export_savedmodel(SAVE_MODEL_DIR, ann_h.no_chestimator_serving_input_reciever)







if __name__ == "__main__":
    inception_modules = []
    tf.app.run(argv=[False])