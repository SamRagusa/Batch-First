import tensorflow as tf

import batch_first.anns.ann_creation_helper as ann_h

from batch_first.chestimator import get_board_data

tf.logging.set_verbosity(tf.logging.INFO)



def score_comparison_model_fn(features, labels, mode, params):
    """
    Generates an EstimatorSpec for a model which scores chess boards.  It learns by maximizing the difference between
    two board evaluations, where one is intended to be greater than the other based on some pre-calculated
    scoring system (e.g. StockFish evaluations).
    """
    convolutional_module_outputs = ann_h.build_convolutional_modules(
        features['board'],
        params['convolutional_modules'],
        mode,
        params['kernel_initializer'],
        params['kernel_regularizer'],
        params['trainable_cnn_modules'])

    if not params["conv_init_fn"] is None:
        params["conv_init_fn"]()

    dense_layers_outputs = ann_h.build_fully_connected_layers_with_batch_norm(
        convolutional_module_outputs,
        params['dense_shape'],
        params['kernel_initializer'],
        mode)

    logits = tf.layers.dense(inputs=dense_layers_outputs,
                             units=1,
                             use_bias=False,
                             activation=None,
                             kernel_initializer=params['kernel_initializer'](),
                             name="logit_layer")

    loss = None
    train_op = None


    # Calculate loss
    if mode != tf.estimator.ModeKeys.PREDICT:
        reshaped_logits = tf.reshape(logits, [-1, 2])

        with tf.variable_scope("loss"):
            evaluation_diff = tf.subtract(*tf.unstack(reshaped_logits, axis=1))

            board_switcher = 2 * tf.cast(features['first_board_greater'], tf.float32) - 1

            evaluation_diff *= board_switcher

            unweighted_loss = -tf.log(tf.sigmoid(evaluation_diff))

            if mode == tf.estimator.ModeKeys.TRAIN:
                loss = tf.reduce_mean(features['loss_weights'] * unweighted_loss)
            else:
                loss = tf.reduce_mean(unweighted_loss)

            loss_summary = tf.summary.scalar("loss", loss)


    # Configure the Training Op
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        learning_rate = params['learning_decay_function'](global_step)
        tf.summary.scalar("learning_rate", learning_rate)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer=params['optimizer'],
            summaries=params['train_summaries'])

    predictions = {"scores" : logits}

    the_export_outputs = {"serving_default" : tf.estimator.export.RegressionOutput(value=logits)}

    if mode != tf.estimator.ModeKeys.PREDICT:
        ratio_better_move_chosen = tf.reduce_mean(tf.cast(tf.greater(evaluation_diff,0), tf.float32))


    validation_metrics = None
    training_hooks = []

    # Create the metrics
    if mode == tf.estimator.ModeKeys.TRAIN:
        to_create_metric_dict = {
            "loss/loss": (loss, loss_summary),
            "loss/unweighted_loss" : unweighted_loss,
            "metrics/mean_evaluation_value" : logits,
            "metrics/mean_abs_evaluation_value": tf.abs(logits),
            "metrics/ratio_better_move_chosen" : tf.cast(ratio_better_move_chosen, tf.float32),
            "metrics/mean_better_minus_worse" : evaluation_diff}

        validation_metrics = ann_h.metric_dict_creator(to_create_metric_dict)

        tf.contrib.layers.summarize_tensors(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        training_hooks.append(
            tf.train.SummarySaverHook(
                save_steps=params['log_interval'],
                output_dir=params['model_dir'],
                summary_op=tf.summary.merge_all()))

    elif mode == tf.estimator.ModeKeys.EVAL:
        validation_metrics = {
            "metrics/mean_better_minus_worse" : tf.metrics.mean(evaluation_diff),
            "metrics/mean_evaluation_value" : tf.metrics.mean(logits),
            "metrics/mean_abs_evaluation_value" : tf.metrics.mean(tf.abs(logits)),
            "metrics/ratio_better_move_chosen" : tf.metrics.mean(tf.cast(ratio_better_move_chosen,tf.float32))}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=training_hooks,
        export_outputs=the_export_outputs,
        eval_metric_ops=validation_metrics)


def simple_board_getter(filename, shuffle_buffer_size, shuffle_seed=None, num_things_in_parallel=None):
    def tf_records_input_data_fn():
        dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=None)

        def parser(record):
            keys_to_features = {"board": tf.FixedLenFeature([8 * 8 ], tf.int64),
                                "score": tf.FixedLenFeature([], tf.int64)}

            parsed_example = tf.parse_single_example(record, keys_to_features)

            return parsed_example['board'], tf.cast(parsed_example['score'], tf.float32)

        dataset = dataset.shuffle(shuffle_buffer_size, shuffle_seed)
        dataset = dataset.map(parser, num_parallel_calls=num_things_in_parallel)
        return dataset

    return tf_records_input_data_fn


def min_max_scaler_creator(start_range=0, end_range=1, clip_min=0, clip_max=10000):
    desired_range = end_range - start_range
    def min_max_scale(tensor):
        clipped = tf.clip_by_value(tensor, clip_min, clip_max)
        min_val = tf.reduce_min(clipped)
        max_val = tf.reduce_max(clipped)

        return start_range + desired_range * (clipped - min_val) / (max_val - min_val)

    return min_max_scale


def score_comparison_input_data_fn(filename_pattern, batch_size, include_unoccupied=True, shuffle_buffer_size=None,
                                   repeat=True, num_things_in_parallel=12, num_things_to_prefetch=None,
                                   shuffle_seeds=(100,250)):

    weight_scaling_fn = min_max_scaler_creator(start_range=.5)

    if num_things_to_prefetch is None:
        num_things_to_prefetch = num_things_in_parallel

    datasets = tuple(
        (simple_board_getter(filename_pattern, shuffle_buffer_size, shuffle_seeds[j], num_things_in_parallel)() for j in
         range(2)))

    combined_dataset = tf.data.Dataset.zip(datasets)

    combined_dataset = combined_dataset.filter(lambda x, y: tf.not_equal(x[1], y[1]))

    combined_dataset = combined_dataset.batch(batch_size, drop_remainder=True)


    def process_batch(x,y):
        stacked_boards = tf.stack([x[0], y[0]], axis=1)

        reshaped_boards = tf.reshape(stacked_boards, [-1, 8, 8])

        omit_unoccupied_decrement = 0 if include_unoccupied else 1
        boards = tf.one_hot(reshaped_boards-omit_unoccupied_decrement, 16-omit_unoccupied_decrement)

        first_greater_second_ints = tf.greater(x[1], y[1])


        scaled_weights = weight_scaling_fn(tf.abs(x[1] - y[1]))

        return boards, first_greater_second_ints, scaled_weights


    combined_dataset = combined_dataset.map(process_batch, num_parallel_calls=10)

    combined_dataset = combined_dataset.prefetch(buffer_size=num_things_to_prefetch)

    if repeat:
        combined_dataset = combined_dataset.repeat()

    iterator = combined_dataset.make_one_shot_iterator()

    features = iterator.get_next()

    feature_dict = {
        "board" : features[0],
        "first_board_greater" : features[1],
        "loss_weights" : features[2]}

    return feature_dict , None


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
    SAVE_MODEL_DIR = "/srv/tmp/sf_comparison_1/pre_commit_test_1"
    TRAINING_FILENAME_PATTERN = "/home/sam/databases/lichess_training.tfrecords"
    VALIDATION_FILENAME_PATTERN = "/home/sam/databases/lichess_validation.tfrecords"
    TRAIN_OP_SUMMARIES = ["gradient_norm", "gradients"]
    NUM_INPUT_FILTERS = 15
    OPTIMIZER = 'Adam'
    TRAINING_SHUFFLE_BUFFER_SIZE = 5000000
    VALIDATION_SHUFFLE_BUFFER_SIZE = 1000
    TRAINING_BATCH_SIZE = 16
    VALIDATION_BATCH_SIZE = 2500
    LOG_ITERATION_INTERVAL = 5000
    LEARNING_RATE = 1e-4
    init_conv_layers_fn = None
    KERNEL_REGULARIZER = lambda: None
    KERNEL_INITIALIZER = lambda: tf.contrib.layers.variance_scaling_initializer()
    MAKE_CNN_MODULES_TRAINABLE = True

    DENSE_SHAPE = []


    # Output of shape: [-1, 8, 8, 170]
    input_module_shape = [[[32, 3]],     # The following 7 layers with 3x3 kernels and increasing dilation factors are
                          [[28, 3, 2]],  # such that their combined filters centered at any given square will consider
                          [[22, 3, 3]],  # only the squares in which a queen could possibly attack from, and the
                          [[18, 3, 4]],  # central square itself (and padding)
                          [[16, 3, 5]],
                          [[12, 3, 6]],
                          [[10, 3, 7]],
                          [[16, 2, (2, 4)]],  # For any given square, this and the following layer collectively consider all possible
                          [[16, 2, (4, 2)]]]  # squares in which a knight could attack from (and aside from padding, no other squares),


    NUM_FINAL_FILTERS = 256

    CONVOLUTIONAL_MODULES = [
        input_module_shape,
        [[[128, 1]] + 6*[[64,3]] + [(NUM_FINAL_FILTERS, 8)]],
        lambda x: tf.reshape(x, [-1, NUM_FINAL_FILTERS]),
    ]

    num_examples_in_training_file = 6926305
    num_examples_in_validation_file = 583324
    num_training_files = 1
    num_validation_files = 1

    BATCHES_IN_TRAINING_EPOCH = (num_training_files * num_examples_in_training_file) // TRAINING_BATCH_SIZE
    BATCHES_IN_VALIDATION_EPOCH = (num_validation_files * num_examples_in_validation_file) // VALIDATION_BATCH_SIZE

    #Currently not using a learning decay
    learning_decay_function = lambda gs : LEARNING_RATE

    WEIGHT_SCALING_FN = min_max_scaler_creator(start_range=.5)

    # Create the Estimator
    the_estimator = tf.estimator.Estimator(
        model_fn=score_comparison_model_fn,
        model_dir=SAVE_MODEL_DIR,
        config=tf.estimator.RunConfig().replace(
            save_checkpoints_steps=LOG_ITERATION_INTERVAL,
            save_summary_steps=LOG_ITERATION_INTERVAL,
            session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.35))),
        params={
            "dense_shape": DENSE_SHAPE,
            "optimizer": OPTIMIZER,
            "log_interval": LOG_ITERATION_INTERVAL,
            "model_dir": SAVE_MODEL_DIR,
            "convolutional_modules" : CONVOLUTIONAL_MODULES,
            "learning_rate": LEARNING_RATE,
            "train_summaries": TRAIN_OP_SUMMARIES,
            "learning_decay_function" : learning_decay_function,
            "num_input_filters" : NUM_INPUT_FILTERS,
            "conv_init_fn": init_conv_layers_fn,
            "kernel_initializer" : KERNEL_INITIALIZER,
            "kernel_regularizer" : KERNEL_REGULARIZER,
            "trainable_cnn_modules" : MAKE_CNN_MODULES_TRAINABLE,
            "weight_scaling_fn" : WEIGHT_SCALING_FN,
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
            shuffle_seeds=(100, 250)),
        temp_num_steps_in_epoch=BATCHES_IN_VALIDATION_EPOCH)

    the_estimator.train(
        input_fn=lambda : score_comparison_input_data_fn(
            TRAINING_FILENAME_PATTERN,
            TRAINING_BATCH_SIZE,
            shuffle_buffer_size=TRAINING_SHUFFLE_BUFFER_SIZE,
            include_unoccupied=NUM_INPUT_FILTERS == 16,
            num_things_in_parallel=None,
            num_things_to_prefetch=1),
        hooks=[validation_hook],
        max_steps=1,
    )

    # Save the model for serving
    the_estimator.export_savedmodel(SAVE_MODEL_DIR, board_eval_serving_input_receiver)





if __name__ == "__main__":
    tf.app.run()