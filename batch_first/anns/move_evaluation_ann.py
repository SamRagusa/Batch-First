import tensorflow as tf

import batch_first.anns.ann_creation_helper as ann_h


tf.logging.set_verbosity(tf.logging.INFO)



def main(using_to_serve):
    """
    Set up the data pipelines, create the computational graph, train the model, and evaluate the results.
    """
    SAVE_MODEL_DIR = "/srv/tmp/move_scoring_3/testing_some_code_6"
    TRAINING_FILENAME_PATTERN = "/srv/databases/chess_engine/sf_chosen_moves_2/move_scoring_training_set_*.tfrecords"
    VALIDATION_FILENAME_PATTERN= "/srv/databases/chess_engine/sf_chosen_moves_2/encoder_training_set_0.tfrecords"
    TRAIN_OP_SUMMARIES = ["gradient_norm", "gradients"]
    NUM_INPUT_FILTERS = 15
    NUM_OUTPUTS = 1792
    DENSE_SHAPE =  [2000,1500,1024,1024,1024]
    OPTIMIZER = 'Adam'
    TRAINING_BATCH_SIZE = 32
    VALIDATION_BATCH_SIZE = 20000
    LOG_ITERATION_INTERVAL = 10000
    LEARNING_RATE = 1e-7
    MAKE_CNN_MODULES_TRAINABLE = True
    # CHECKPOINT_DIR_WITH_CONV_LAYERS = "/srv/tmp/current/small_cnn_2"
    KERNEL_INITIALIZER = lambda: tf.contrib.layers.variance_scaling_initializer()
    KERNEL_REGULARIZER = lambda: None
    init_conv_layers_fn = None


    INCEPTION_MODULES = [
        [
            [[15, 2], [20, 2]],
            [[15, 3]]],
        [
            [[25, 1]],
            [[15, 1], [25, 1, 6]],
            [[15, 1], [25, 6, 1]]]]

    BATCHES_IN_TRAINING_EPOCH = (5*2009392) // (TRAINING_BATCH_SIZE)
    BATCHES_IN_VALIDATION_EPOCH = 2009392 // VALIDATION_BATCH_SIZE


    learning_decay_function = lambda gs  : tf.train.exponential_decay(LEARNING_RATE,
                                                                      global_step=gs,
                                                                      decay_steps=BATCHES_IN_TRAINING_EPOCH,
                                                                      decay_rate=0.94,
                                                                      staircase=True)


    # scopes_to_restore = ["inception_module_1_path_1", "inception_module_1_path_2","inception_module_2_path_3", "inception_module_2_path_2", "inception_module_2_path_3", "inception_module_4_path_1"]
    # dict_things_to_restore = {name + "/": name + "/" for name in scopes_to_restore}
    #
    # init_conv_layers_fn = lambda: tf.train.init_from_checkpoint(CHECKPOINT_DIR_WITH_CONV_LAYERS, dict_things_to_restore)


    # Create the Estimator
    the_estimator = tf.estimator.Estimator(
        model_fn=ann_h.move_gen_cnn_model_fn,
        model_dir=SAVE_MODEL_DIR,
        config=tf.estimator.RunConfig().replace(
            save_checkpoints_steps=LOG_ITERATION_INTERVAL,
            save_summary_steps=LOG_ITERATION_INTERVAL,
            session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.1))),
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
            "kernel_initializer": KERNEL_INITIALIZER,
            "kernel_regularizer" : KERNEL_REGULARIZER,
        })


    validation_hook = ann_h.ValidationRunHook(
        step_increment=BATCHES_IN_TRAINING_EPOCH//5,
        estimator=the_estimator,
        input_fn_creator=lambda: ann_h.move_gen_create_tf_records_input_data_fn(
            VALIDATION_FILENAME_PATTERN,
            VALIDATION_BATCH_SIZE,
            include_unoccupied=NUM_INPUT_FILTERS==16,
            repeat=False,
            shuffle=False),
        temp_num_steps_in_epoch=BATCHES_IN_VALIDATION_EPOCH,
        recall_input_fn_creator_after_evaluate=True)



    the_estimator.train(
        input_fn=ann_h.move_gen_create_tf_records_input_data_fn(
            TRAINING_FILENAME_PATTERN,
            TRAINING_BATCH_SIZE,
            include_unoccupied=NUM_INPUT_FILTERS==16),
        hooks=[validation_hook],
        # max_steps=1,
    )

    # Save the model for serving
    the_estimator.export_savedmodel(SAVE_MODEL_DIR, ann_h.no_chestimator_serving_move_scoring_input_reciever)





if __name__ == "__main__":
    tf.app.run(argv=[False])