import tensorflow as tf

from scipy.stats import kendalltau, weightedtau, spearmanr

import batch_first.anns.ann_creation_helper as ann_h

tf.logging.set_verbosity(tf.logging.INFO)



def diag_comparison_model_fn(features, labels, mode, params):
    """
    Generates an EstimatorSpec for a model which scores chess boards.  It learns by maximizing the difference between
    board evaluation values, where one is intended to be greater than the other based on some pre-calculated
    scoring system (e.g. StockFish evaluations).
    """
    convolutional_module_outputs=ann_h.create_input_convolutions_shared_weights(
        features['board'],
        params['kernel_initializer'],
        params['data_format'],
        mode,
        num_unique_filters=[32, 20, 20, 18, 18, 16, 24, 24]) #236 with the 64 undilated included

    if len(params['convolutional_modules']):
        convolutional_module_outputs = ann_h.build_convolutional_modules(
            convolutional_module_outputs,
            params['convolutional_modules'],
            mode,
            params['kernel_initializer'],
            params['kernel_regularizer'],
            params['trainable_cnn_modules'],
            num_previous_modules=1,
            data_format=params['data_format'])

    logits = tf.layers.conv2d(
        inputs=convolutional_module_outputs,
        filters=1,
        kernel_size=[1,1],
        padding="valid",
        data_format="channels_last" if params['data_format']=="NHWC" else "channels_first",
        use_bias=False,
        kernel_initializer=params['kernel_initializer'](),
        kernel_regularizer=params['kernel_regularizer'](),
        name="logit_layer")

    logits = tf.squeeze(logits, axis=[1,2,3])

    loss = None
    train_op = None


    # Calculate loss
    if mode != tf.estimator.ModeKeys.PREDICT:
        with tf.variable_scope("loss"):
            calculated_diff_matrix = ann_h.vec_and_transpose_op(logits, tf.subtract)

            label_matrix = features['label_matrix']
            weight_matrix = features['weight_matrix']

            loss = tf.losses.sigmoid_cross_entropy(label_matrix, calculated_diff_matrix, weights=weight_matrix)
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

    validation_metrics = None
    training_hooks = []

    # Create the metrics
    if mode == tf.estimator.ModeKeys.TRAIN:
        tau_a = ann_h.kendall_rank_correlation_coefficient(logits, features['score'])

        to_create_metric_dict = {
            "loss/loss": (loss, loss_summary),
            "metrics/mean_evaluation_value" : logits,
            "metrics/mean_abs_evaluation_value": tf.abs(logits),
            "metrics/kendall_tau-a" : tau_a,
        }

        validation_metrics = ann_h.metric_dict_creator(to_create_metric_dict)

        tf.contrib.layers.summarize_tensors(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        training_hooks.append(
            tf.train.SummarySaverHook(
                save_steps=params['log_interval'],
                output_dir=params['model_dir'],
                summary_op=tf.summary.merge_all()))

    elif mode == tf.estimator.ModeKeys.EVAL:
        the_logits, update1 = tf.contrib.metrics.streaming_concat(logits)
        the_labels, update2 = tf.contrib.metrics.streaming_concat(features['score'])

        BIGGER_THAN_CP_LESS_THAN_WIN_VAL = tf.constant(100000, dtype=features['score'].dtype)

        mate_mask = tf.greater(tf.abs(the_labels), BIGGER_THAN_CP_LESS_THAN_WIN_VAL)
        mate_logits = tf.boolean_mask(the_logits, mate_mask)
        mate_labels = tf.boolean_mask(the_labels, mate_mask)

        non_mate_mask = tf.logical_not(mate_mask)
        non_mate_logits = tf.boolean_mask(the_logits, non_mate_mask)
        non_mate_labels = tf.boolean_mask(the_labels, non_mate_mask)

        mate_tau_b, mate_tau_b_p_value = ann_h.py_func_scipy_rank_helper_creator(mate_logits, mate_labels)(kendalltau)

        non_mate_coef_creator = ann_h.py_func_scipy_rank_helper_creator(non_mate_logits, non_mate_labels)
        non_mate_tau_b, non_mate_tau_b_p_value = non_mate_coef_creator(kendalltau)
        non_mate_weighted_tau_b, non_mate_weighted_tau_b_p_value= non_mate_coef_creator(weightedtau)

        update = tf.group(update1, update2)

        scipy_rank_coef_creator = ann_h.py_func_scipy_rank_helper_creator(the_logits, the_labels)

        tau_b, tau_b_p_value = scipy_rank_coef_creator(kendalltau)
        rho, rho_p_value = scipy_rank_coef_creator(spearmanr)

        validation_metrics = {
            "metrics/mean_evaluation_value" : tf.metrics.mean(logits),
            "metrics/mean_abs_evaluation_value" : tf.metrics.mean(tf.abs(logits)),
            "metrics/kendall_tau-b": (tau_b, update),
            "metrics/kendall_tau-b_p_value": (tau_b_p_value, update),
            "metrics/kendall_tau-b_mate_only" : (mate_tau_b, update),
            "metrics/non_mate_kendall_tau-b" : (non_mate_tau_b, update),
            "metrics/non_mate_weighted_tau-b": (non_mate_weighted_tau_b, update),
            "metrics/non_mate_weighted_kendall_tau-b_p_value": (non_mate_weighted_tau_b_p_value, update),
            "metrics/spearman_rho": (rho, update),
            "metrics/spearman_rho_p_value": (rho_p_value, update),
        }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=training_hooks,
        export_outputs=the_export_outputs,
        eval_metric_ops=validation_metrics)


def lower_diag_score_comparison_input_fn(filename_pattern, batch_size, include_unoccupied=True, shuffle_buffer_size=None,
                                         num_things_in_parallel=None, num_things_to_prefetch=None, shuffle_seed=None,
                                         data_format="NHWC"):
    if num_things_to_prefetch is None:
        num_things_to_prefetch = tf.contrib.data.AUTOTUNE  #IMPORTANT NOTE: This seems to make the program crash after a few iterations (at least it does on my computer)

    dataset = tf.data.TFRecordDataset(filename_pattern)

    if shuffle_buffer_size:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(shuffle_buffer_size,seed=shuffle_seed))

    dataset = dataset.batch(batch_size, drop_remainder=True)

    def process_batch(records):
        keys_to_features = {"board": tf.FixedLenFeature([8 * 8], tf.int64),
                            "score": tf.FixedLenFeature([], tf.int64)}

        parsed_examples = tf.parse_example(records, keys_to_features)

        reshaped_boards = tf.reshape(parsed_examples['board'], [-1, 8, 8])

        omit_unoccupied_decrement = 0 if include_unoccupied else 1
        boards = tf.one_hot(
            reshaped_boards - omit_unoccupied_decrement,
            16 - omit_unoccupied_decrement,
            axis=-1 if data_format=="NHWC" else 1)

        desired_diff_matrix = ann_h.vec_and_transpose_op(parsed_examples['score'], tf.subtract, tf.float32)

        lower_diag_diff_matrix = tf.matrix_band_part(desired_diff_matrix, -1, 0)

        lower_diag_sign = tf.sign(lower_diag_diff_matrix)
        weight_mask = tf.abs(lower_diag_sign)
        bool_weight_mask = tf.cast(weight_mask, tf.bool)

        comparison_indices = tf.where(bool_weight_mask)

        # value_larger_than_centipawn_less_than_mate = 100000
        # desired_found_mate = tf.greater(tf.abs(parsed_examples['score']), value_larger_than_centipawn_less_than_mate)

        # both_found_mate = ann_h.vec_and_transpose_op(desired_found_mate, tf.logical_and)

        # desired_signs = tf.sign(parsed_examples['score'])

        # same_sign_matrix = ann_h.vec_and_transpose_op(desired_signs, tf.equal)

        # both_same_player_mates = tf.logical_and(both_found_mate, same_sign_matrix)

        # both_same_mate_and_nonzero_weight = tf.logical_and(both_same_player_mates, bool_weight_mask)

        # same_mate_depth_diff_decrement = .95

        # weight_helper = same_mate_depth_diff_decrement * tf.cast(both_same_mate_and_nonzero_weight, tf.float32)

        # mate_adjusted_weight_mask = weight_mask - weight_helper

        label_matrix = (lower_diag_sign + weight_mask)/2

        # return boards, parsed_examples['score'], label_matrix, mate_adjusted_weight_mask, comparison_indices
        return boards, parsed_examples['score'], label_matrix, weight_mask, comparison_indices


    dataset = dataset.map(process_batch, num_parallel_calls=num_things_in_parallel)

    dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0', buffer_size=num_things_to_prefetch))

    iterator = dataset.make_one_shot_iterator()

    features = iterator.get_next()

    feature_dict = {"board": features[0], "score": features[1], "label_matrix": features[2], "weight_matrix": features[3]}

    return feature_dict, None


def board_eval_serving_input_receiver(data_format="NCHW"):
    def fn_to_return():
        placeholder_shape = [None, 15, 8, 8] if data_format=="NCHW" else [None, 8, 8, 15]

        for_remapping = tf.placeholder(tf.float32, placeholder_shape, "FOR_INPUT_MAPPING_transpose")

        receiver_tensors = {"board": for_remapping}
        return tf.estimator.export.ServingInputReceiver(receiver_tensors, receiver_tensors)
    return fn_to_return


def main(unused_par):

    SAVE_MODEL_DIR = "/srv/tmp/diag_loss_3/pre_commit_test_2534111"
    TRAINING_FILENAME_PATTERN = "/srv/databases/lichess_combined_methods_eval_databases/lichess_training.tfrecords"
    VALIDATION_FILENAME_PATTERN = "/srv/databases/lichess_combined_methods_eval_databases/lichess_validation.tfrecords"
    TRAIN_OP_SUMMARIES = ["gradient_norm", "gradients"]
    NUM_INPUT_FILTERS = 15
    OPTIMIZER = 'Adam'
    TRAINING_SHUFFLE_BUFFER_SIZE = 16800000
    TRAINING_BATCH_SIZE = 512      #The effective batch size used for the loss = n(n-1)/2  (where n is the number of boards in the batch)
    VALIDATION_BATCH_SIZE = 1000
    LOG_ITERATION_INTERVAL = 2500
    LEARNING_RATE = 2.5e-3
    KERNEL_REGULARIZER = lambda: None
    KERNEL_INITIALIZER = lambda: tf.contrib.layers.variance_scaling_initializer()
    TRAINABLE_CNN_MODULES = True
    DATA_FORMAT = "NCHW"
    SAME_MATE_DEPTH_DIFF_LOSS_WEIGHT_DECREMENT = .95
    VALUE_LARGER_THAN_CENTIPAWN_LESS_THAN_MATE = 100000

    num_examples_in_training_file = 16851682
    num_examples_in_validation_file = 1982551

    BATCHES_IN_TRAINING_EPOCH = num_examples_in_training_file // TRAINING_BATCH_SIZE
    BATCHES_IN_VALIDATION_EPOCH =  num_examples_in_validation_file // VALIDATION_BATCH_SIZE

    # learning_decay_function = lambda gs: LEARNING_RATE
    learning_decay_function = lambda gs : tf.train.exponential_decay(LEARNING_RATE, gs,
                                                                     BATCHES_IN_TRAINING_EPOCH, 0.96, staircase=True)

    CONVOLUTIONAL_MODULES = [[[[512, 1], [128, 1]] + 6 * [[32, 3]] + [(16, 8)]]]


    # Create the Estimator
    the_estimator = tf.estimator.Estimator(
        model_fn=diag_comparison_model_fn,
        model_dir=SAVE_MODEL_DIR,
        config=tf.estimator.RunConfig().replace(
            save_checkpoints_steps=LOG_ITERATION_INTERVAL,
            save_summary_steps=LOG_ITERATION_INTERVAL),
        params={
            "optimizer": OPTIMIZER,
            "log_interval": LOG_ITERATION_INTERVAL,
            "model_dir": SAVE_MODEL_DIR,
            "convolutional_modules" : CONVOLUTIONAL_MODULES,
            "learning_rate": LEARNING_RATE,
            "train_summaries": TRAIN_OP_SUMMARIES,
            "learning_decay_function" : learning_decay_function,
            "num_input_filters" : NUM_INPUT_FILTERS,
            "kernel_initializer" : KERNEL_INITIALIZER,
            "kernel_regularizer" : KERNEL_REGULARIZER,
            "trainable_cnn_modules" : TRAINABLE_CNN_MODULES,
            "same_mate_depth_diff_decrement" : SAME_MATE_DEPTH_DIFF_LOSS_WEIGHT_DECREMENT,
            "value_larger_than_centipawn_less_than_mate" : VALUE_LARGER_THAN_CENTIPAWN_LESS_THAN_MATE,
            "data_format" : DATA_FORMAT,
            })


    validation_hook = ann_h.ValidationRunHook(
        step_increment=BATCHES_IN_TRAINING_EPOCH,
        estimator=the_estimator,
        input_fn_creator=lambda: lambda : lower_diag_score_comparison_input_fn(
            VALIDATION_FILENAME_PATTERN,
            VALIDATION_BATCH_SIZE,
            include_unoccupied=NUM_INPUT_FILTERS == 16,
            num_things_to_prefetch=5,
            num_things_in_parallel=12,
            data_format=DATA_FORMAT,
        ),
        temp_num_steps_in_epoch=BATCHES_IN_VALIDATION_EPOCH)

    the_estimator.train(
        input_fn=lambda : lower_diag_score_comparison_input_fn(
            TRAINING_FILENAME_PATTERN,
            TRAINING_BATCH_SIZE,
            shuffle_buffer_size=TRAINING_SHUFFLE_BUFFER_SIZE,
            include_unoccupied=NUM_INPUT_FILTERS == 16,
            num_things_in_parallel=12,
            num_things_to_prefetch=36,
            data_format=DATA_FORMAT,
            shuffle_seed=12312312,
        ),
        hooks=[validation_hook],
        # max_steps=1,
    )

    # Save the model for inference
    the_estimator.export_savedmodel(SAVE_MODEL_DIR, board_eval_serving_input_receiver())




if __name__ == "__main__":
    tf.app.run()
