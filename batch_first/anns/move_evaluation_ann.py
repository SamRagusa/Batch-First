import tensorflow as tf

import batch_first.anns.ann_creation_helper as ann_h

from batch_first.chestimator import get_board_data

tf.logging.set_verbosity(tf.logging.INFO)


def all_conv_move_gen_cnn_model_fn(features, labels, mode, params):
    """
    Generates an EstimatorSpec for the model.
    """
    inception_module_outputs = ann_h.build_convolutional_modules(
        features["board"],
        params['inception_modules'],
        mode,
        params['kernel_initializer'],
        params['kernel_regularizer'],
        params['trainable_cnn_modules'])


    logits = tf.layers.conv2d(
        inputs=inception_module_outputs,
        filters=params['num_outputs'],
        kernel_size=1,
        padding='valid',
        use_bias=False,
        kernel_initializer=params['kernel_initializer'](),
        kernel_regularizer=params['kernel_regularizer'](),
        name="logit_layer")


    loss = None
    train_op = None

    reshaped_logits = tf.reshape(logits, (-1, 64, params['num_outputs']))
    legal_move_logits = tf.gather_nd(reshaped_logits, features["legal_move_indices"])

    # Compute loss
    if mode != tf.estimator.ModeKeys.PREDICT:
        with tf.variable_scope("loss"):
            # loss = tf.reduce_mean(tf.log(tf.cosh(features["desired_scores"] - legal_move_logits)))
            loss = tf.losses.huber_loss(features["desired_scores"], legal_move_logits)
            # loss = tf.losses.mean_squared_error(features["desired_scores"], legal_move_logits, weights=.1)
            loss_scalar_summary = tf.summary.scalar("loss", loss)



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


    # Generate predictions
    predictions = {"the_move_values": inception_module_outputs}

    # A dictionary for scoring used when exporting model for serving.
    the_export_outputs = {"serving_default": tf.estimator.export.ClassificationOutput(scores=legal_move_logits)}


    # Create the metrics
    validation_metrics = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        calculated_best_move_scores = tf.gather(legal_move_logits, features['desired_move_indices'])

        repeated_best_scores = ann_h.numpy_style_repeat_1d(calculated_best_move_scores, features['num_moves'])

        move_is_below_best = tf.cast(tf.greater_equal(repeated_best_scores, legal_move_logits), dtype=tf.float32)

        diff_from_desired = legal_move_logits - features["desired_scores"]

        abs_diff_from_desired = tf.abs(diff_from_desired)

        if mode == tf.estimator.ModeKeys.EVAL:
            validation_metrics = {
                "metrics/ratio_moves_below_best" : tf.metrics.mean(move_is_below_best),
                "metrics/distance_from_desired" : tf.metrics.mean(diff_from_desired),
                "metrics/abs_distance_from_desired" : tf.metrics.mean(abs_diff_from_desired),
            }
        else:
            mean_calculated_value = tf.reduce_mean(legal_move_logits)
            mean_diff_from_desired = tf.reduce_mean(diff_from_desired
                                                    )
            to_create_metric_dict = {
                "loss/loss": (loss, loss_scalar_summary),
                "metrics/ratio_moves_below_best": tf.reduce_mean(move_is_below_best),
                "metrics/mean_evaluation_value": mean_calculated_value,
                "metrics/mean_abs_evaluation_value": tf.abs(legal_move_logits),
                "metrics/mean_expected_value": features["desired_scores"],
                "metrics/mean_abs_expected_value": abs(features["desired_scores"]),
                "metrics/distance_from_desired": mean_diff_from_desired,
                "metrics/abs_distance_from_desired": abs_diff_from_desired,
                "metrics/relative_distance_from_desired": tf.abs(mean_diff_from_desired / mean_calculated_value),
            }

            validation_metrics = ann_h.metric_dict_creator(to_create_metric_dict)



    tf.contrib.layers.summarize_tensors(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    summary_hook = tf.train.SummarySaverHook(save_steps=params['log_interval'],
                                             output_dir=params['model_dir'],
                                             summary_op=tf.summary.merge_all())

    # Return the EstimatorSpec object
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=[summary_hook],
        export_outputs=the_export_outputs,
        eval_metric_ops=validation_metrics)


def move_index_getter(from_squares, to_squares, moves_per_board):
    to_repeat = tf.range(tf.shape(moves_per_board)[0])
    board_indices_for_moves = ann_h.numpy_style_repeat_1d(to_repeat, moves_per_board)

    if from_squares.dtype != board_indices_for_moves:
        board_indices_for_moves = tf.cast(board_indices_for_moves, from_squares.dtype)

    return tf.stack([board_indices_for_moves, from_squares, to_squares], axis=-1)


def move_scoring_serving_input_receiver():
    (piece_bbs, color_occupied_bbs, ep_squares, castling_lookup_indices, kings), formatted_data = get_board_data()

    moves_per_board = tf.placeholder(tf.uint8, shape=[None], name="moves_per_board_placeholder")
    moves = tf.placeholder(tf.uint8, shape=[None, 2], name="move_placeholder")

    unpacked_moves = tf.unstack(tf.cast(moves, tf.int32), axis=1)

    the_moves = move_index_getter(unpacked_moves[0], unpacked_moves[1], moves_per_board)

    receiver_tensors = {"piece_bbs" : piece_bbs,
                        "color_occupied_bbs" : color_occupied_bbs,
                        "ep_squares" : ep_squares,
                        "castling_lookup_indices": castling_lookup_indices,
                        "kings": kings,
                        "moves_per_board" : moves_per_board,
                        "moves" : moves}

    dict_for_model_fn = {"board" : formatted_data,
                         "legal_move_indices" : the_moves}

    return tf.estimator.export.ServingInputReceiver(dict_for_model_fn , receiver_tensors)


def move_scoring_create_tf_records_input_data_fn(filename_pattern, batch_size, shuffle_buffer_size=50000,
                                                 include_unoccupied=True, repeat=True, shuffle=True,
                                                 num_things_in_parallel=1, num_files=1):

    def input_fn():
        filenames = tf.data.Dataset.list_files(filename_pattern)

        dataset = filenames.interleave(
            lambda filename : tf.data.TFRecordDataset(filename),
            cycle_length=num_files)

        def parser(records):
            features = {
                "board": tf.FixedLenFeature([64], tf.int64),
                "from_squares" : tf.VarLenFeature(tf.int64),
                "to_squares": tf.VarLenFeature(tf.int64),
                "move_scores": tf.VarLenFeature(tf.float32),
                "num_moves" : tf.FixedLenFeature([], tf.int64)
            }

            parsed_record = tf.parse_example(records, features)

            move_lookup_indices = move_index_getter(
                parsed_record["from_squares"].values,
                parsed_record["to_squares"].values,
                parsed_record['num_moves'])

            reshaped_board = tf.reshape(parsed_record["board"], [-1, 8, 8])

            if include_unoccupied:
                one_hot_board = tf.one_hot(reshaped_board, 16)
            else:
                one_hot_board = tf.one_hot(reshaped_board - 1, 15)

            dense_desired_moves = tf.sparse_tensor_to_dense(parsed_record['move_scores'],default_value=tf.float32.min)

            most_desired_moves = tf.argmax(dense_desired_moves, axis=1)

            adjusted_desired_moves = most_desired_moves +  tf.cumsum(parsed_record['num_moves'], exclusive=True)

            move_scores = parsed_record['move_scores'].values

            return one_hot_board, move_lookup_indices, move_scores, adjusted_desired_moves, parsed_record['num_moves']


        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

        dataset = dataset.batch(batch_size)

        dataset = dataset.map(parser, num_parallel_calls=num_things_in_parallel)

        dataset = dataset.prefetch(num_things_in_parallel)

        if repeat:
            dataset = dataset.repeat()

        iterator = dataset.make_one_shot_iterator()

        features = iterator.get_next()

        feature_names = ["board", "legal_move_indices", "desired_scores", "desired_move_indices", "num_moves"]

        for_model = dict(zip(feature_names, features))

        return for_model, None

    return input_fn



def main(unused_par):
    """
    Set up the data pipelines, create the computational graph, train the model, and evaluate the results.
    """
    SAVE_MODEL_DIR = "/srv/tmp/move_scoring_4/to_from_square_3"
    TRAINING_FILENAME_PATTERN = "/srv/databases/chess_engine/move_scoring_2/move_scoring_training_set_*.tfrecords"
    VALIDATION_FILENAME_PATTERN = "/srv/databases/chess_engine/move_scoring_2/move_scoring_validation_set_*.tfrecords"
    TRAIN_OP_SUMMARIES = ["gradient_norm", "gradients"]
    NUM_INPUT_FILTERS = 15
    NUM_OUTPUTS = 64  #73
    OPTIMIZER = 'Adam'
    TRAINING_BATCH_SIZE = 32
    VALIDATION_BATCH_SIZE = 5000
    LOG_ITERATION_INTERVAL = 5000
    LEARNING_RATE = 1e-7
    MAKE_CNN_MODULES_TRAINABLE = True
    KERNEL_INITIALIZER = lambda: tf.contrib.layers.variance_scaling_initializer()
    KERNEL_REGULARIZER = lambda: None
    init_conv_layers_fn = None




    # Output of shape: [-1, 8, 8, 126]
    input_module_shape = [[[20, 3]],           #The following 7 diluted 3x3 layers are such that the
                          [[20, 3, 2]],        #filters centered at any given square will look at every square
                          [[20, 3, 3]],        #where a queen could possibly attack from, and the
                          [[20, 3, 4]],        #centered square itself (and aside from padding, no other squares)
                          [[10, 3, 5]],
                          [[10, 3, 6]],
                          [[10, 3, 7]],
                          [[8, 2, (2, 4)]],    #For any given square, this and the following layer collectivly look at all possible
                          [[8, 2, (4, 2)]]]    #squares in which a knight could attack from (and aside from padding, no other squares)



    # Output of shape: [-1, 8, 8, 115]
    repeated_module_shape = [[[30, 1]],
                             [[15, 1], [30, 3]],
                             [[12, 1], [20, 3, 2]],
                             [[12, 1], [20, 3, 3]],
                             [[10, 1], [15, 3, 4]]]



    INCEPTION_MODULES = [
        input_module_shape,
        repeated_module_shape,
        repeated_module_shape,
    ]


    num_in_one_file = 1133357
    num_training_files = 8
    num_validation_files = 2

    BATCHES_IN_TRAINING_EPOCH = (num_training_files * num_in_one_file) // TRAINING_BATCH_SIZE
    BATCHES_IN_VALIDATION_EPOCH = (num_validation_files * num_in_one_file) // VALIDATION_BATCH_SIZE

    LEARNING_DECAY_FN = lambda gs: LEARNING_RATE   #Not currently using a decay


    # scopes_to_restore = ["inception_module_1_path_1", "inception_module_1_path_2","inception_module_2_path_3", "inception_module_2_path_2", "inception_module_2_path_3", "inception_module_4_path_1"]
    # dict_things_to_restore = {name + "/": name + "/" for name in scopes_to_restore}
    #
    # init_conv_layers_fn = lambda: tf.train.init_from_checkpoint(CHECKPOINT_DIR_WITH_CONV_LAYERS, dict_things_to_restore)


    # Create the Estimator
    the_estimator = tf.estimator.Estimator(
        model_fn=all_conv_move_gen_cnn_model_fn,
        model_dir=SAVE_MODEL_DIR,
        config=tf.estimator.RunConfig().replace(
            save_checkpoints_steps=LOG_ITERATION_INTERVAL,
            save_summary_steps=LOG_ITERATION_INTERVAL,
            session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.35))),
        params={
            "optimizer": OPTIMIZER,
            "num_outputs": NUM_OUTPUTS,
            "log_interval": LOG_ITERATION_INTERVAL,
            "model_dir": SAVE_MODEL_DIR,
            "inception_modules" : INCEPTION_MODULES,
            "learning_rate": LEARNING_RATE,
            "train_summaries": TRAIN_OP_SUMMARIES,
            "learning_decay_function" : LEARNING_DECAY_FN,
            "trainable_cnn_modules" : MAKE_CNN_MODULES_TRAINABLE,
            "conv_init_fn" : init_conv_layers_fn,
            "kernel_initializer": KERNEL_INITIALIZER,
            "kernel_regularizer" : KERNEL_REGULARIZER,
        })


    validation_hook = ann_h.ValidationRunHook(
        step_increment=BATCHES_IN_TRAINING_EPOCH//5,
        estimator=the_estimator,
        input_fn_creator=lambda: move_scoring_create_tf_records_input_data_fn(
            VALIDATION_FILENAME_PATTERN,
            VALIDATION_BATCH_SIZE,
            include_unoccupied=NUM_INPUT_FILTERS==16,
            num_things_in_parallel=1,
            repeat=False,
            shuffle=False,
            num_files=num_validation_files),
        temp_num_steps_in_epoch=BATCHES_IN_VALIDATION_EPOCH,
        recall_input_fn_creator_after_evaluate=True)

    the_estimator.train(
        input_fn=move_scoring_create_tf_records_input_data_fn(
            TRAINING_FILENAME_PATTERN,
            TRAINING_BATCH_SIZE,
            include_unoccupied=NUM_INPUT_FILTERS==16,
            num_things_in_parallel=1,
            num_files=num_training_files),
        hooks=[validation_hook],
        # max_steps=1,
    )

    # Save the model for serving
    the_estimator.export_savedmodel(SAVE_MODEL_DIR, move_scoring_serving_input_receiver)





if __name__ == "__main__":
    tf.app.run()