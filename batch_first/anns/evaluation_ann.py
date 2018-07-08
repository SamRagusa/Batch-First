'''
Created on Jul 2, 2017

@author: SamRagusa
'''

import tensorflow as tf

import batch_first.anns.ann_creation_helper as ann_h

tf.logging.set_verbosity(tf.logging.INFO)




def main(unused_par):
    """
    Set up the data pipelines, create the computational graph, train the model, and evaluate the results.
    """
    SAVE_MODEL_DIR = "/srv/tmp/sf_mean_loss/comparison_triple_module_2"#comparison_double_module_2"
    TRAINING_FILENAME_PATTERN = "/home/sam/databases/scoring_training_set_*.tfrecords"
    VALIDATION_FILENAME_PATTERN = "/home/sam/databases/scoring_validation_set_*.tfrecords"
    TRAIN_OP_SUMMARIES = ["gradient_norm", "gradients"]
    NUM_INPUT_FILTERS = 15
    NUM_OUTPUTS = 1
    OPTIMIZER = 'Adam'
    TRAINING_SHUFFLE_BUFFER_SIZE = 500000
    VALIDATION_SHUFFLE_BUFFER_SIZE = 25000
    TRAINING_BATCH_SIZE = 32
    VALIDATION_BATCH_SIZE = 10000
    LOG_ITERATION_INTERVAL = 10000
    LEARNING_RATE = 1e-4
    init_conv_layers_fn = None
    KERNEL_REGULARIZER = lambda: None
    KERNEL_INITIALIZER = lambda: tf.contrib.layers.variance_scaling_initializer()
    MAKE_CNN_MODULES_TRAINABLE = True

    DENSE_SHAPE = [1024, 512, 512]


    INCEPTION_MODULES = [
        [[[15, 2]],
         [[25, 3]],
         [[20, 1, 8]],
         [[20, 8, 1]]],  # Output of shape: [-1, 8, 8, 80]

        [[[25, 1]],
         [[15, 1], [25, 3]],
         [[10, 1], [20, 1, 8]],
         [[10, 1], [20, 8, 1]]],  # Module output of shape: [-1, 8, 8, 90]

        [[[20, 1]],  # 8*8*20 = 1280
         [[15, 1], (25, 3)],  # 6*6*25 = 900
         [[15, 1], (50, 1, 8)],  # 8*1*50 = 400
         [[15, 1], (50, 8, 1)]],  # 1*8*50 = 400      Module output after_flattening shape: [-1, 3300]
    ]




    num_in_one_file = 1133357#2171817
    num_training_files = 5
    num_validation_files = 3

    BATCHES_IN_TRAINING_EPOCH = (num_training_files * 2171817) // TRAINING_BATCH_SIZE#num_in_one_file) // TRAINING_BATCH_SIZE
    BATCHES_IN_VALIDATION_EPOCH = (num_validation_files * num_in_one_file) // VALIDATION_BATCH_SIZE


    learning_decay_function = lambda gs  : tf.train.exponential_decay(LEARNING_RATE,
                                                                      global_step=gs,
                                                                      decay_steps=BATCHES_IN_TRAINING_EPOCH,
                                                                      decay_rate=0.96,
                                                                      staircase=True)


    # Create the Estimator
    the_estimator = tf.estimator.Estimator(
        model_fn=ann_h.score_comparison_model_fn,
        model_dir=SAVE_MODEL_DIR,
        config=tf.estimator.RunConfig().replace(
            save_checkpoints_steps=LOG_ITERATION_INTERVAL,
            save_summary_steps=LOG_ITERATION_INTERVAL,
            session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.35))),
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
            "kernel_regularizer" : KERNEL_REGULARIZER,
            "trainable_cnn_modules" : MAKE_CNN_MODULES_TRAINABLE,
            })


    validation_hook = ann_h.ValidationRunHook(
        step_increment=BATCHES_IN_TRAINING_EPOCH//2,
        estimator=the_estimator,
        input_fn_creator=lambda: lambda : ann_h.score_comparison_input_data_fn(
            VALIDATION_FILENAME_PATTERN,
            VALIDATION_BATCH_SIZE,
            shuffle_buffer_size=VALIDATION_SHUFFLE_BUFFER_SIZE,
            include_unoccupied=NUM_INPUT_FILTERS == 16,
            num_things_to_prefetch=1),
        temp_num_steps_in_epoch=BATCHES_IN_VALIDATION_EPOCH)

    the_estimator.train(
        input_fn=lambda : ann_h.score_comparison_input_data_fn(
            TRAINING_FILENAME_PATTERN,
            TRAINING_BATCH_SIZE,
            shuffle_buffer_size=TRAINING_SHUFFLE_BUFFER_SIZE,
            include_unoccupied=NUM_INPUT_FILTERS == 16),
        hooks=[validation_hook],
        # max_steps=1,
    )

    # Save the model for serving
    the_estimator.export_savedmodel(SAVE_MODEL_DIR, ann_h.board_eval_serving_input_receiver)





if __name__ == "__main__":
    tf.app.run()