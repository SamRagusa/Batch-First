'''
Created on Jul 2, 2017

@author: SamRagusa
'''

import tensorflow as tf

import batch_first.anns.ann_creation_helper as ann_h

from batch_first.chestimator import get_board_data


tf.logging.set_verbosity(tf.logging.INFO)



def score_comparison_model_fn(features, labels, mode, params):
    """
    Generates an EstimatorSpec for the model.
    """

    if mode == tf.estimator.ModeKeys.PREDICT:
        input_layer = features['board']
    else:
        # Reshape features from original shape of [-1, 2, 8, 8, params['num_input_filters']]
        input_layer = tf.reshape(features, [-1, 8, 8, params['num_input_filters']])


    inception_module_outputs = ann_h.build_convolutional_modules(
        input_layer,
        params['inception_modules'],
        mode,
        params['kernel_initializer'],
        params['kernel_regularizer'],
        params['trainable_cnn_modules'])


    if not params["conv_init_fn"] is None:
        params["conv_init_fn"]()


    # Build the fully connected layers
    dense_layers_outputs = ann_h.build_fully_connected_layers_with_batch_norm(
        inception_module_outputs,
        params['dense_shape'],
        params['kernel_initializer'],
        mode)

    # Create the final layer of the ANN
    logits = tf.layers.dense(inputs=dense_layers_outputs,
                             units=params['num_outputs'],
                             use_bias=False,
                             activation=None,
                             kernel_initializer=params['kernel_initializer'](),
                             name="logit_layer")

    loss = None
    train_op = None


    # Calculate loss (for both TRAIN and EVAL modes)
    if mode != tf.estimator.ModeKeys.PREDICT:
        reshaped_logits = tf.reshape(logits, [-1, 2])

        first_boards_logits, second_boards_logits = tf.split(reshaped_logits, 2, axis=1)
        first_boards_labels, second_boards_labels = tf.split(labels, 2, axis=1)

        first_board_intended_greater = tf.greater(first_boards_labels, second_boards_labels)

        first_logits_minus_second = first_boards_logits - second_boards_logits

        swapped_inequalities_logits_diff = first_logits_minus_second * (2*tf.cast(first_board_intended_greater, tf.float32) - 1)

        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(-tf.log(tf.sigmoid(swapped_inequalities_logits_diff)))
            loss_summary = tf.summary.scalar("loss", loss)


    # Configure the Training Op
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        learning_rate = params['learning_decay_function'](global_step)
        # tf.summary.scalar("learning_rate", learning_rate)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer=params['optimizer'],
            summaries=params['train_summaries'])


    # Generate predictions
    predictions = {"scores" : logits}

    # A dictionary for scoring used when exporting model for serving.
    the_export_outputs = {"serving_default" : tf.estimator.export.RegressionOutput(value=logits)}


    if mode != tf.estimator.ModeKeys.PREDICT:
        ratio_better_move_chosen = tf.greater(swapped_inequalities_logits_diff, 0)


    validation_metrics = None
    training_hooks = []

    # Create the metrics
    if mode == tf.estimator.ModeKeys.TRAIN:
        to_create_metric_dict = {
            "loss/loss": (loss, loss_summary),
            "metrics/mean_evaluation_value" : logits,
            "metrics/mean_abs_evaluation_value": tf.abs(logits),
            "metrics/ratio_better_move_chosen" : tf.cast(ratio_better_move_chosen, tf.float32),
            "metrics/mean_better_minus_worse" : swapped_inequalities_logits_diff,
        }

        validation_metrics = ann_h.metric_dict_creator(to_create_metric_dict)

        tf.contrib.layers.summarize_tensors(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        training_hooks = [tf.train.SummarySaverHook(save_steps=params['log_interval'],
                                                    output_dir=params['model_dir'],
                                                    summary_op=tf.summary.merge_all())]

    elif mode == tf.estimator.ModeKeys.EVAL:
        abs_logits = tf.abs(logits)

        validation_metrics = {
            "metrics/mean_better_minus_worse" : tf.metrics.mean(swapped_inequalities_logits_diff),
            "metrics/mean_evaluation_value" : tf.metrics.mean(logits),
            "metrics/mean_abs_evaluation_value" : tf.metrics.mean(abs_logits),
            "metrics/ratio_better_move_chosen" : tf.metrics.mean(tf.cast(ratio_better_move_chosen,tf.float32)),
        }


    # Return the EstimatorSpec object
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=training_hooks,
        export_outputs=the_export_outputs,
        eval_metric_ops=validation_metrics)


def one_hot_create_tf_records_input_data_fn(filename_pattern, batch_size=None, include_unoccupied=True,
                                            shuffle_buffer_size=None, repeat=True, num_things_in_parallel=12,
                                            return_before_prefetch=False, shuffle_seed=None, num_files=1):
    def tf_records_input_data_fn():
        filenames = tf.data.Dataset.list_files(filename_pattern)

        dataset = filenames.interleave(
            lambda filename : tf.data.TFRecordDataset(filename),
            cycle_length=num_files)

        def parser(record):
            keys_to_features = {
                "board": tf.FixedLenFeature([8 * 8 ], tf.int64),
                "score": tf.FixedLenFeature([], tf.int64),
            }

            parsed_example = tf.parse_single_example(record, keys_to_features)

            reshaped_board = tf.reshape(parsed_example['board'], [8,8])

            if include_unoccupied:
                one_hot_board = tf.one_hot(reshaped_board, 16)
            else:
                one_hot_board = tf.one_hot(reshaped_board - 1, 15)

            return one_hot_board, tf.cast(parsed_example['score'], tf.float32)

        if not shuffle_buffer_size is None:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=shuffle_seed)

        if not batch_size is None:
            dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parser, batch_size=batch_size, num_parallel_batches=num_things_in_parallel))
        else:
            dataset = dataset.map(parser, num_parallel_calls=num_things_in_parallel)

        if return_before_prefetch:
            return dataset

        dataset = dataset.prefetch(buffer_size=num_things_in_parallel)

        if repeat:
            dataset = dataset.repeat()

        iterator = dataset.make_one_shot_iterator()

        features = iterator.get_next()

        for_model = {"board" : features[0]}

        return for_model, features[1]

    return tf_records_input_data_fn


def score_comparison_input_data_fn(filename_pattern, batch_size, include_unoccupied=True,
                                   shuffle_buffer_size=None, repeat=True, num_things_in_parallel=12, num_things_to_prefetch=None, num_files=1, shuffle_seeds=(100,250)):

    if num_things_to_prefetch is None:
        num_things_to_prefetch = num_things_in_parallel



    dataset_1 = one_hot_create_tf_records_input_data_fn(filename_pattern, None, include_unoccupied,
                                                        shuffle_buffer_size, repeat, num_things_in_parallel,
                                                        return_before_prefetch=True, shuffle_seed=shuffle_seeds[0], num_files=num_files)()

    dataset_2 = one_hot_create_tf_records_input_data_fn(filename_pattern, None, include_unoccupied,
                                                        shuffle_buffer_size, repeat, num_things_in_parallel,
                                                        return_before_prefetch=True, shuffle_seed=shuffle_seeds[1], num_files=num_files)()


    combined_dataset = tf.data.Dataset.zip((dataset_1, dataset_2))

    combined_dataset = combined_dataset.filter(lambda x, y: x[1] != y[1])

    combined_dataset = combined_dataset.apply(
        tf.contrib.data.map_and_batch(
            map_func=lambda x, y: (tf.stack([x[0], y[0]]), tf.stack([x[1], y[1]])),
            batch_size=batch_size,
            num_parallel_batches=num_things_in_parallel))


    combined_dataset = combined_dataset.prefetch(buffer_size=num_things_to_prefetch)

    if repeat:
        combined_dataset = combined_dataset.repeat()

    iterator = combined_dataset.make_one_shot_iterator()

    features = iterator.get_next()

    return features[0], features[1]


def board_eval_serving_input_receiver():
    (piece_bbs, color_occupied_bbs, ep_squares, castling_lookup_indices, kings), formatted_data = get_board_data()

    receiver_tensors = {"piece_bbs": piece_bbs,
                        "color_occupied_bbs": color_occupied_bbs,
                        "ep_squares": ep_squares,
                        "castling_lookup_indices": castling_lookup_indices,
                        "kings": kings}

    dict_for_model_fn = {"board": formatted_data}

    return tf.estimator.export.ServingInputReceiver(dict_for_model_fn, receiver_tensors)



def main(unused_par):
    """
    Set up the data pipelines, create the computational graph, train the model, and evaluate the results.
    """
    SAVE_MODEL_DIR = "/srv/tmp/sf_comparison/no_modules_5"
    TRAINING_FILENAME_PATTERN = "/home/sam/databases/scoring_training_set_*.tfrecords"
    VALIDATION_FILENAME_PATTERN = "/home/sam/databases/scoring_validation_set_*.tfrecords"
    TRAIN_OP_SUMMARIES = ["gradient_norm", "gradients"]
    NUM_INPUT_FILTERS = 15
    NUM_OUTPUTS = 1
    OPTIMIZER = 'Adam'
    TRAINING_SHUFFLE_BUFFER_SIZE = 50000
    VALIDATION_SHUFFLE_BUFFER_SIZE = 1000
    TRAINING_BATCH_SIZE = 32
    VALIDATION_BATCH_SIZE = 10000
    LOG_ITERATION_INTERVAL = 5000
    LEARNING_RATE = 1e-6
    init_conv_layers_fn = None
    KERNEL_REGULARIZER = lambda: None
    KERNEL_INITIALIZER = lambda: tf.contrib.layers.variance_scaling_initializer()
    MAKE_CNN_MODULES_TRAINABLE = True

    DENSE_SHAPE = [512, 256, 128]


    # Output of shape: [-1, 8, 8, 126]
    input_module_shape = [[[20, 3]],       # The following 7 layers with 3x3 kernels and increasing dilution rates are
                          [[20, 3, 2]],    # such that their combined filters centered at any given square will consider
                          [[20, 3, 3]],    # only the squares in which a queen could possibly attack from, and the
                          [[20, 3, 4]],    # central square itself (and padding)
                          [[10, 3, 5]],
                          [[10, 3, 6]],
                          [[10, 3, 7]],
                          [[8, 2, (2, 4)]],     # For any given square, this and the following layer collectively consider all possible
                          [[8, 2, (4, 2)]]]     # squares in which a knight could attack from (and aside from padding, no other squares),


    INCEPTION_MODULES = [
        input_module_shape,

        [[[32,1], (1024, 8)]],
        lambda x: tf.reshape(x, [-1, 1024]),
    ]


    num_in_one_file = 1133357
    num_training_files = 10
    num_validation_files = 2

    BATCHES_IN_TRAINING_EPOCH = (num_training_files * num_in_one_file) // TRAINING_BATCH_SIZE
    BATCHES_IN_VALIDATION_EPOCH = (num_validation_files * num_in_one_file) // VALIDATION_BATCH_SIZE


    learning_decay_function = lambda gs : LEARNING_RATE


    # Create the Estimator
    the_estimator = tf.estimator.Estimator(
        model_fn=score_comparison_model_fn,
        model_dir=SAVE_MODEL_DIR,
        config=tf.estimator.RunConfig().replace(
            save_checkpoints_steps=LOG_ITERATION_INTERVAL,
            save_summary_steps=LOG_ITERATION_INTERVAL,
            session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.7))),
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
        step_increment=BATCHES_IN_TRAINING_EPOCH//4,
        estimator=the_estimator,
        input_fn_creator=lambda: lambda : score_comparison_input_data_fn(
            VALIDATION_FILENAME_PATTERN,
            VALIDATION_BATCH_SIZE,
            shuffle_buffer_size=VALIDATION_SHUFFLE_BUFFER_SIZE,
            repeat=False,
            include_unoccupied=NUM_INPUT_FILTERS == 16,
            num_things_to_prefetch=12,
            num_files=num_validation_files),
        temp_num_steps_in_epoch=BATCHES_IN_VALIDATION_EPOCH)

    the_estimator.train(
        input_fn=lambda : score_comparison_input_data_fn(
            TRAINING_FILENAME_PATTERN,
            TRAINING_BATCH_SIZE,
            shuffle_buffer_size=TRAINING_SHUFFLE_BUFFER_SIZE,
            include_unoccupied=NUM_INPUT_FILTERS == 16,
            num_files=num_training_files,
            shuffle_seeds=(1337,42)), # The newest implementation relies on beimg given shuffling random seeds, but I
                                      # don't believe that currently the boards being compared will change over
                                      # multiple epochs. This WILL be fixed/addressed in the next commit.
        hooks=[validation_hook],
        # max_steps=1,
    )

    # Save the model for serving
    the_estimator.export_savedmodel(SAVE_MODEL_DIR, board_eval_serving_input_receiver)





if __name__ == "__main__":
    tf.app.run()