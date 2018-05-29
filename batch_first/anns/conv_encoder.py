import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.estimator.estimator import Estimator

import ann_creation_helper as ann_h



tf.logging.set_verbosity(tf.logging.INFO)







def main(using_to_serve):
    """
    Set up the data pipelines, create the computational graph, train the model, and evaluate the results.
    """
    SAVE_MODEL_DIR = "/srv/tmp/encoder/pre_commit_test"#with_moves_7_regularized"
    TRAINING_FILENAME_PATTERN = "/srv/databases/chess_engine/move_scoring_1/move_scoring_training_set_*.tfrecords"
    VALIDATION_FILENAME_PATTERN = "/srv/databases/chess_engine/move_scoring_1/move_scoring_validation_set_*.tfrecords"
    # TESTING_FILENAME_PATTERN = "/srv/databases/chess_engine/move_scoring_1/move_scoring_testing_set_*.tfrecords"
    TRAIN_OP_SUMMARIES = ["gradient_norm", "gradients"]
    NUM_INPUT_FILTERS = 15
    # NUM_OUTPUTS = 1792
    OPTIMIZER = 'Adam'
    TRAINING_BATCH_SIZE = 64#200#500
    VALIDATION_BATCH_SIZE = 20000
    # TESTING_BATCH_SIZE = 10000
    LOG_ITERATION_INTERVAL = 10000
    LEARNING_RATE = 1e-3#.000001#.000002
    MAKE_CNN_MODULES_TRAINABLE = True
    init_conv_layers_fn = None
    noise_factor = .5
    WEIGHT_REGULARIZATION_FN = lambda : layers.l2_regularizer(noise_factor)#lambda:None
    LABEL_TENSOR_NAME = "Reshape:0"#"inception_module_3_path_1/Relu:0"
    KERNEL_INITIALIZER = lambda : layers.variance_scaling_initializer(factor=1,mode='FAN_IN', uniform=True)


    ENCODER_MODULE = [[[20, 3],  # 720
                       [35, 3],  # 560
                       [400, 4]]]  # 400


    DECODER_MODULE = [[[100, 2, 1], #400
                       [45, 3, 1], #800
                       [55, 3, 1]]] #2160  with final layer having 80*8*8=5120



    INCEPTION_MODULES = [
        ENCODER_MODULE,


        lambda tensor: ann_h.build_transposed_inception_module_with_batch_norm(tensor,
                                                                               DECODER_MODULE,
                                                                               kernel_initializer=KERNEL_INITIALIZER,
                                                                               mode=tf.estimator.ModeKeys.TRAIN,
                                                                               padding="valid",
                                                                               weight_regularizer=WEIGHT_REGULARIZATION_FN,
                                                                               num_previously_built_inception_modules=1)[0],


        lambda tensor: tf.layers.conv2d_transpose(tensor, 80, 3, strides=1,
                                                  kernel_initializer=KERNEL_INITIALIZER(),
                                                  kernel_regularizer=WEIGHT_REGULARIZATION_FN(),
                                                  padding="valid",
                                                  activation=None,
                                                  use_bias=False),
    ]



    BATCHES_IN_TRAINING_EPOCH = (7*2009392) // (TRAINING_BATCH_SIZE)
    BATCHES_IN_VALIDATION_EPOCH = 2009392 // VALIDATION_BATCH_SIZE


    learning_decay_function = lambda gs  : tf.train.exponential_decay(LEARNING_RATE,
                                                                      global_step=gs,
                                                                      decay_steps=BATCHES_IN_TRAINING_EPOCH,#25,
                                                                      decay_rate=0.96,
                                                                      staircase=True)




    # Create the Estimator
    the_estimator = Estimator(
        model_fn=ann_h.encoder_builder_fn,
        model_dir=SAVE_MODEL_DIR,
        config=tf.estimator.RunConfig().replace(
            save_checkpoints_steps=LOG_ITERATION_INTERVAL,
            save_summary_steps=LOG_ITERATION_INTERVAL,
            session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.3))),
        params={
            "optimizer": OPTIMIZER,
            "log_interval": LOG_ITERATION_INTERVAL,
            "model_dir": SAVE_MODEL_DIR,
            "inception_modules" : INCEPTION_MODULES,
            "learning_rate": LEARNING_RATE,
            "train_summaries": TRAIN_OP_SUMMARIES,
            "learning_decay_function" : learning_decay_function,
            "trainable_cnn_modules" : MAKE_CNN_MODULES_TRAINABLE,
            "conv_init_fn" : init_conv_layers_fn,
            "num_input_filters": NUM_INPUT_FILTERS,
            "kernel_regularizer" : WEIGHT_REGULARIZATION_FN,
            "label_tensor_name" : LABEL_TENSOR_NAME,
            "kernel_initializer" : KERNEL_INITIALIZER,
        })


    validation_hook = ann_h.ValidationRunHook(
        step_increment=BATCHES_IN_TRAINING_EPOCH,
        estimator=the_estimator,
        input_fn_creator=lambda: ann_h.encoder_tf_records_input_data_fn(
            VALIDATION_FILENAME_PATTERN,
            VALIDATION_BATCH_SIZE,
            include_unoccupied=NUM_INPUT_FILTERS==16,
            repeat=False,
            shuffle=False),
        temp_num_steps_in_epoch=BATCHES_IN_VALIDATION_EPOCH,
        recall_input_fn_creator_after_evaluate=True)


    the_estimator.train(
        input_fn=ann_h.encoder_tf_records_input_data_fn(
            TRAINING_FILENAME_PATTERN,
            TRAINING_BATCH_SIZE,
            shuffle_buffer_size=50000,
            include_unoccupied=NUM_INPUT_FILTERS==16),
        hooks=[validation_hook],
        # max_steps=1,
    )






if __name__ == "__main__":
    tf.app.run(argv=[False])