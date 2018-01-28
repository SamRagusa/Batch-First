'''
Created on Oct 10, 2017
@author: SamRagusa
'''


import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from tensorflow.python import training

from functools import reduce




def build_fully_connected_layers_with_batch_norm(the_input, shape, mode, num_previous_fully_connected_layers=0, activation_summaries=[], scope_prefix=""):
    """
    a function to build the fully connected layers with batch normalization onto the computational
    graph from given specifications.

    shape of the format:
    [num_neurons_layer_1,num_neurons_layer_2,...,num_neurons_layer_n]
    """

    for index, size in enumerate(shape):
        with tf.variable_scope(scope_prefix + "FC_" + str(num_previous_fully_connected_layers + index + 1)):
            temp_pre_activation = tf.layers.dense(
                inputs=the_input,
                units=size,
                use_bias=False,
                kernel_initializer=layers.xavier_initializer(),
                name="layer")

            temp_batch_normalized = tf.layers.batch_normalization(temp_pre_activation,
                                                                  training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                                  fused=True)

            temp_layer_output = tf.nn.relu(temp_batch_normalized)

            the_input = temp_layer_output

        activation_summaries.append(layers.summarize_activation(temp_layer_output))

    return the_input, activation_summaries


def build_inception_module_with_batch_norm(the_input, module, mode, activation_summaries=[], num_previously_built_inception_modules=0, padding='same', force_no_concat=False,make_trainable=True):
    """
    NOTE:
    1) This comment no longer fully describes the functionality of the function.  It will be updated in the near future
    when I have a bit more time to focus on this type of stuff.

    Builds an inception module based on the design given to the function.  It returns the final layer in the module,
    and the activation summaries generated for the layers within the inception module.

    The layers will be named "module_N_path_M/layer_P", where N is the inception module number, M is what path number it is on,
    and P is what number layer it is in that path.

    Module of the format:
    [[[filters1_1,kernal_size1_1],... , [filters1_M,kernal_size1_M]],... ,
        [filtersN_1,kernal_sizeN_1],... , [filtersN_P,kernal_sizeN_P]]
    """
    path_outputs = [None for _ in range(len(module))]
    to_summarize = []
    cur_input = None
    for j, path in enumerate(module):
        with tf.variable_scope("inception_module_" + str(num_previously_built_inception_modules + 1) + "_path_" + str(j + 1)):
            for i, section in enumerate(path):
                if i == 0:
                    if j != 0:
                        path_outputs[j - 1] = cur_input

                    cur_input = the_input
                kernel_size = [section[1], section[1]] if len(section)==2 else [section[1], section[2]]
                cur_conv_output = tf.layers.conv2d(
                    inputs=cur_input,
                    filters=section[0],
                    kernel_size=kernel_size,
                    padding=padding,
                    use_bias=False,
                    kernel_initializer=layers.xavier_initializer(),
                    trainable=make_trainable,
                    name="layer_" + str(i + 1))

                cur_batch_normalized = tf.layers.batch_normalization(cur_conv_output,
                                                                     training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                                     trainable=make_trainable,
                                                                     fused=True)

                cur_input = tf.nn.relu(cur_batch_normalized)

                to_summarize.append(cur_input)

    path_outputs[-1] = cur_input

    activation_summaries = activation_summaries + [layers.summarize_activation(layer) for layer in to_summarize]

    with tf.variable_scope("inception_module_" + str(num_previously_built_inception_modules + 1)):
        for j in range(1, len(path_outputs)):
            if force_no_concat or path_outputs[0].get_shape().as_list()[1:3] != path_outputs[j].get_shape().as_list()[1:3]:
                return [temp_input for temp_input in path_outputs], activation_summaries

        return tf.concat([temp_input for temp_input in path_outputs], 3), activation_summaries


def mean_metric_creator(name):
    def mean_metric(inputs, forced_labels_param, weights=None):
        mean_value = tf.reduce_mean(inputs)
        return mean_value, tf.summary.scalar(name, mean_value)

    return mean_metric


def cnn_model_fn(features, labels, mode, params):
    """
    Generates an EstimatorSpec for the model.
    """

    if mode == tf.estimator.ModeKeys.PREDICT:
        input_layer = features["feature"]
    else:
        #Reshape features from original shape of [-1, 3, 8, 8, 16]
        input_layer = tf.reshape(features, [-1, 8, 8, 16])

    cur_inception_module = input_layer


    for module_num, module_shape in enumerate(params['inception_modules']):
        cur_inception_module, activation_summaries = build_inception_module_with_batch_norm(
            cur_inception_module,
            module_shape,
            mode,
            padding='valid',
            num_previously_built_inception_modules=module_num)

    if isinstance(cur_inception_module, list):
        inception_module_paths_flattened = [
            tf.reshape(
                path,
                [-1, reduce(lambda a, b: a * b, path.get_shape().as_list()[1:])]
            ) for path in cur_inception_module]

        inception_module_outputs = tf.concat(inception_module_paths_flattened, 1)
    else:
        #I think this causes problems if it occurs (not happening in current model)
        inception_module_outputs = cur_inception_module


    # Build the fully connected layers
    dense_layers_outputs, activation_summaries = build_fully_connected_layers_with_batch_norm(
        inception_module_outputs,
        params['dense_shape'],
        mode,
        activation_summaries=activation_summaries)


    # Create the final layer of the ANN
    logits = tf.layers.dense(inputs=dense_layers_outputs,
                             units=params['num_outputs'],
                             use_bias=False,
                             activation=None,
                             kernel_initializer=layers.xavier_initializer(),
                             name="logit_layer")

    loss = None
    train_op = None
    ratio_old_new_sum_loss_to_negative_sum = None

    # (p,q,r) = (original_position, choosen_position, random_position)
    original_pos, desired_pos, random_pos = tf.split(tf.reshape(logits, [-1, 3, 1]), [1, 1, 1], 1)

    # Calculate loss (for both TRAIN and EVAL modes)
    if mode != tf.estimator.ModeKeys.PREDICT:

        # Implementing  an altered version of the loss function defined in Deep Pink
        with tf.variable_scope("loss"):
            increase_in_position_value = tf.constant(params["equality_scalar"], dtype=tf.float32)

            adjusted_equality_sum = tf.scalar_mul(increase_in_position_value, original_pos) + desired_pos


            real_greater_rand_scalar_loss = tf.reduce_mean(-tf.log(tf.sigmoid(desired_pos - random_pos)))

            # old_new_squared_scalar_loss = tf.reduce_mean(tf.square(original_pos + desired_pos))
            # loss = real_greater_rand_scalar_loss +  old_new_squared_scalar_loss


            equality_scalar_loss = tf.reduce_mean(-tf.log(tf.sigmoid(adjusted_equality_sum)))
            negative_equality_scalar_loss = tf.reduce_mean(-tf.log(tf.sigmoid(tf.negative(adjusted_equality_sum))))

            ratio_old_new_sum_loss_to_negative_sum = tf.divide(equality_scalar_loss, negative_equality_scalar_loss)

            real_rand_loss_summary = tf.summary.scalar("real_greater_rand_loss", real_greater_rand_scalar_loss)


            equality_sum_loss_summary = tf.summary.scalar("mean_original_plus_desired_loss", equality_scalar_loss)
            negative_equality_sum_loss_summary = tf.summary.scalar("mean_negative_original_plus_desired", negative_equality_scalar_loss)

            loss = real_greater_rand_scalar_loss + equality_scalar_loss + negative_equality_scalar_loss


            # tf.summary.scalar("old_real_squared_loss", old_new_squared_scalar_loss)
            tf.summary.scalar("loss", loss)


    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        learning_rate = params['learning_decay_function'](global_step)
        tf.summary.scalar("learning_rate", learning_rate)
        train_op = layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer=params['optimizer'],
            summaries=params['train_summaries'])


    # Generate predictions (should be None, but it doesn't allow for that)
    # predictions = None
    # if mode != tf.estimator.ModeKeys.PREDICT:
    predictions = {
        "real_rand_guessing": tf.greater(desired_pos, random_pos)}

    # A dictionary for scoring used when exporting model for serving.
    the_export_outputs = {
        "serving_default" : tf.estimator.export.RegressionOutput(value=logits)}

    real_rand_diff = desired_pos - random_pos
    old_plus_desired = original_pos + desired_pos
    # Create the validation metrics
    validation_metric = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        validation_metric = {
            "metrics/rand_vs_real_accuracy": mean_metric_creator("metrics/rand_vs_real_accuracy")(tf.cast(tf.greater(desired_pos, random_pos), tf.float32), None),
            "metrics/mean_dist_real_rand": mean_metric_creator("metrics/mean_dist_real_rand")(real_rand_diff, None),
            "metrics/mean_abs_dist_real_rand": mean_metric_creator("metrics/mean_abs_dist_real_rand")(tf.abs(real_rand_diff), None),
            "metrics/mean_dist_old_real": mean_metric_creator("metrics/mean_dist_old_real")(old_plus_desired, None),
            "metrics/mean_abs_dist_old_real": mean_metric_creator("metrics/mean_abs_dist_old_real")(tf.abs(old_plus_desired), None),
            "loss/real_greater_rand_loss": (real_greater_rand_scalar_loss, real_rand_loss_summary),
            "loss/mean_original_plus_desired_loss": (equality_scalar_loss, equality_sum_loss_summary),
            "loss/mean_negative_original_plus_desired": (negative_equality_scalar_loss, negative_equality_sum_loss_summary),
            "loss/ratio_old_new_sum_loss_to_negative_sum" : (ratio_old_new_sum_loss_to_negative_sum, tf.summary.scalar("loss/ratio_old_new_sum_loss_to_negative_sum", ratio_old_new_sum_loss_to_negative_sum)),
            # "sanity_check/input_tensor_mean" : mean_metric_creator("sanity_check/input_tensor_mean")(features, None),
            "metrics/mean_old_pos": mean_metric_creator("metrics/mean_old_pos")(original_pos, None),
            "metrics/mean_new_pos": mean_metric_creator("metrics/mean_new_pos")(desired_pos, None),
            "metrics/negative_new_to_old_ratio" : mean_metric_creator("metrics/negative_new_to_old_ratio")(tf.divide(-desired_pos, original_pos), None),
            "metrics/mean_random_pos": mean_metric_creator("metrics/mean_random_pos")(random_pos, None),
            "metrics/mean_abs_old_pos": mean_metric_creator("metrics/mean_abs_old_pos")(tf.abs(original_pos), None),
            "metrics/mean_abs_new_pos": mean_metric_creator("metrics/mean_abs_new_pos")(tf.abs(desired_pos), None),
            "metrics/mean_abs_random_pos": mean_metric_creator("metrics/mean_abs_random_pos")(tf.abs(random_pos), None)}

    # Create the trainable variable summaries and merge them together to give to a hook
    trainable_var_summaries = layers.summarize_tensors(tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES))  # Not sure if needs to be stored as a variable, should check
    merged_summaries = tf.summary.merge_all()
    summary_hook = tf.train.SummarySaverHook(save_steps=params['log_interval'],
                                             output_dir=params['model_dir'],
                                             summary_op=merged_summaries)

    # Return the EstimatorSpec object
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=[summary_hook],
        export_outputs=the_export_outputs,
        eval_metric_ops=validation_metric)


def move_gen_cnn_model_fn(features, labels, mode, params):
    """
    Generates an EstimatorSpec for the model.
    """

    def index_tensor_to_index_pairs(index_tensor):
        """
        Takes an array of indices and returns an array defined by the following (not perfectly descriptive
        but you get the idea):
        output[j] = [j,index_matrix[j]]
        """
        replicated_first_indices = tf.tile(
            tf.expand_dims(tf.range(tf.shape(index_tensor, out_type=tf.int64)[0], dtype=tf.int64), dim=1),
            [1, tf.shape(index_tensor)[1]])
        return tf.stack([replicated_first_indices, index_tensor], axis=2)



    if mode == tf.estimator.ModeKeys.PREDICT:
        input_layer = features["feature"]
    else:
        input_layer = features

    cur_inception_module = input_layer
    for module_num, module_shape in enumerate(params['inception_modules']):
        cur_inception_module, activation_summaries = build_inception_module_with_batch_norm(
            cur_inception_module,
            module_shape,
            mode,
            padding='valid',
            num_previously_built_inception_modules=module_num,
            make_trainable=params["trainable_cnn_modules"])

    if isinstance(cur_inception_module, list):
        inception_module_paths_flattened = [
            tf.reshape(
                path,
                [-1, reduce(lambda a, b: a * b, path.get_shape().as_list()[1:])]
            ) for path in cur_inception_module]

        inception_module_outputs = tf.concat(inception_module_paths_flattened, 1)
    else:
        inception_module_outputs = cur_inception_module



    # Build the fully connected layers
    dense_layers_outputs, activation_summaries = build_fully_connected_layers_with_batch_norm(
        inception_module_outputs,
        params['dense_shape'],
        mode,
        activation_summaries=activation_summaries)


    # Create the final layer of the ANN
    logits = tf.layers.dense(inputs=dense_layers_outputs,
                             units=params['num_outputs'],
                             use_bias=False,
                             activation=None,
                             kernel_initializer=layers.xavier_initializer(),
                             name="logit_layer")



    loss = None
    train_op = None

    bool_mask = None


    # Calculate loss (for both TRAIN and EVAL modes)
    if mode != tf.estimator.ModeKeys.PREDICT:
        weight_mask = labels

        bool_mask = tf.cast(weight_mask, tf.bool)
        weight_mask = tf.to_float(bool_mask)

        opposite_bool_mask = tf.logical_not(bool_mask)
        important_logits = tf.boolean_mask(logits, bool_mask)
        important_labels = tf.boolean_mask(labels, bool_mask)

        mean_possible_moves = tf.reduce_mean(weight_mask)
        total_possible_moves = tf.reduce_sum(weight_mask)




        min_float_tensor = tf.fill(tf.shape(labels), np.finfo(np.float32).min)

        labels_with_illegal_moves_set_to_min_value = tf.where(bool_mask, labels, min_float_tensor)
        logits_with_illegal_moves_set_to_min_value = tf.where(bool_mask, logits, min_float_tensor)

        indices_of_desired_moves = tf.argmax(labels_with_illegal_moves_set_to_min_value, axis=1)

        index_pairs = index_tensor_to_index_pairs(tf.expand_dims(indices_of_desired_moves, axis=1))

        max_legal_calculated_move_values = tf.gather_nd(logits, index_pairs)

        move_scored_higher_than_desired_move = tf.greater_equal(logits_with_illegal_moves_set_to_min_value,
                                                                max_legal_calculated_move_values)

        total_moves_above_desired_moves = tf.reduce_sum(tf.cast(move_scored_higher_than_desired_move, dtype=tf.float32))


        with tf.variable_scope("loss"):

            loss = tf.losses.mean_squared_error(labels, logits, weights=weight_mask)
            tf.summary.scalar("loss", loss)



    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        learning_rate = params['learning_decay_function'](global_step)
        tf.summary.scalar("learning_rate", learning_rate)
        train_op = layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer=params['optimizer'],
            summaries=params['train_summaries'])


    # Generate predictions
    predictions = {
        "move_values": logits}

    # A dictionary for scoring used when exporting model for serving.
    the_export_outputs = {
        # "serving_default" : tf.estimator.export.PredictOutput(outputs={"logits": logits}),
        "serving_default" : tf.estimator.export.ClassificationOutput(scores=logits),
    }



    # Create the validation metrics
    validation_metric = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        validation_metric = {
            "metrics/total_moves_above_desired_moves" : (total_moves_above_desired_moves, tf.summary.scalar("metrics/total_moves_above_desired_moves", total_moves_above_desired_moves)),
            "metrics/ratio_moves_above_desired_moves": (total_moves_above_desired_moves/total_possible_moves,tf.summary.scalar("metrics/ratio_moves_above_desired_moves",total_moves_above_desired_moves/total_possible_moves)),
            "sanity_check/mean_possible_moves" : (mean_possible_moves, tf.summary.scalar("sanity_check/mean_possible_moves", mean_possible_moves)),
            "sanity_check/total_possible_moves" : (total_possible_moves, tf.summary.scalar("sanity_check/total_possible_moves", total_possible_moves)),
            "metrics/mean_evaluation_value" : mean_metric_creator("metrics/mean_evaluation_value")(important_logits,None),
            "metrics/mean_expected_value" : mean_metric_creator("metrics/mean_expected_value")(important_labels,None),
            "metrics/mean_abs_expected_value": mean_metric_creator("metrics/mean_abs_expected_value")(abs(important_labels), None),
            "metrics/distance_from_desired" : mean_metric_creator("metrics/distance_from_desired")(important_logits-important_labels,None),
            "metrics/abs_distance_from_desired": mean_metric_creator("metrics/abs_distance_from_desired")(tf.abs(important_logits - important_labels), None),
            "metrics/distance_from_not_desired": mean_metric_creator("metrics/distance_from_not_desired")(tf.boolean_mask(logits, opposite_bool_mask) - tf.boolean_mask(labels, opposite_bool_mask),None),
            "metrics/relative_distance_from_desired": mean_metric_creator("metrics/relative_distance_from_desired")(tf.abs((important_logits - tf.boolean_mask(labels, bool_mask))/important_logits), None),
        }

    # Create the trainable variable summaries and merge them together to give to a hook
    trainable_var_summaries = layers.summarize_tensors(tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES))  # Not sure if needs to be stored as a variable, should check
    merged_summaries = tf.summary.merge_all()
    summary_hook = tf.train.SummarySaverHook(save_steps=params['log_interval'],
                                             output_dir=params['model_dir'],
                                             summary_op=merged_summaries)

    # Return the EstimatorSpec object
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=[summary_hook],
        export_outputs=the_export_outputs,
        eval_metric_ops=validation_metric)


def process_line_as_2d_input_with_ep(the_str):
    """
    NOTES:
    1) I likely won't be using this, opting to instead use the onehot implementation
    """
    with tf.name_scope("process_data_2d"):
        #     with tf.device("/cpu:0"):

        # A tensor referenced when getting indices of characters for the the_values array
        mapping_strings = tf.constant(["0", "1", "K", "Q", "R", "B", "N", "P", "C", "k", "q", "r", "b", "n", "p", "c"])

        the_values = tf.constant(
            [[0, 0, 0, 0, 0, 0, 0, 0],  # 0
             [0, 0, 0, 0, 0, 0, 1, 0],  # 1
             [1, 0, 0, 0, 0, 0, 0, 0],  # K
             [0, 1, 0, 0, 0, 0, 0, 0],  # Q
             [0, 0, 1, 0, 0, 0, 0, 0],  # R
             [0, 0, 0, 1, 0, 0, 0, 0],  # B
             [0, 0, 0, 0, 1, 0, 0, 0],  # N
             [0, 0, 0, 0, 0, 1, 0, 0],  # P
             [0, 0, 0, 0, 0, 0, 0, 1],  # C
             [-1, 0, 0, 0, 0, 0, 0, 0],  # k
             [0, -1, 0, 0, 0, 0, 0, 0],  # q
             [0, 0, -1, 0, 0, 0, 0, 0],  # r
             [0, 0, 0, -1, 0, 0, 0, 0],  # b
             [0, 0, 0, 0, -1, 0, 0, 0],  # n
             [0, 0, 0, 0, 0, -1, 0, 0],  # p
             [0, 0, 0, 0, 0, 0, 0, -1],  # c
             ], dtype=tf.float32)

        # Create the table for getting indices (for the_values) from the information about the board
        the_table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings, name="index_lookup_table")

        data = tf.reshape(
            # Get the values at the given indices
            tf.gather(
                the_values,
                # Get an array of indices corresponding to the array of characters
                the_table.lookup(
                    # Split the string into an array of characters
                    tf.string_split(
                        [the_str],
                        delimiter="").values)),
            [3, 64, 8])

        return data


def full_onehot_process_line_as_2d_input(the_str, num_samples=-1, for_serving=False):
    with tf.name_scope("process_data_2d"):
        #with tf.device("/cpu:0"):

        # A tensor referenced when getting indices of characters for the the_values array
        mapping_strings = tf.constant(
            ["0", "1", "K", "Q", "R", "B", "N", "P", "C", "k", "q", "r", "b", "n", "p", "c"])

        number_of_mapping_strings = 16  # len(mapping_strings)
        the_values = tf.constant(
            [[1 if i == j else 0 for i in range(number_of_mapping_strings)] for j in range(number_of_mapping_strings)],
            dtype=tf.float32)

        # Create the table for getting indices (for the_values) from the information about the board
        the_table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings, name="index_lookup_table")

        value_for_string_split = the_str if for_serving else [the_str]

        data = tf.reshape(
            # Get the values at the given indices
            tf.gather(
                the_values,
                # Get an array of indices corresponding to the array of characters
                the_table.lookup(
                    # Split the string into an array of characters
                    tf.string_split(
                        value_for_string_split,
                        delimiter="").values)),
            [num_samples, 8,8, number_of_mapping_strings]) #THIS SHOULD REALLY BE [3x8x8,num_mapping_strings]

        return data


def acquire_data_ops(filename_queue, processing_method, record_defaults=None):
    """
    Get the line/lines from the files in the given filename queue,
    read/decode them, and give them to the given method for processing
    the information.
    """
    with tf.name_scope("acquire_data"):
        # with tf.device("/cpu:0"):
        if record_defaults is None:
            record_defaults = [[""]]
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)
        row = tf.decode_csv(value, record_defaults=record_defaults)
        #The 3 is because this is used for training and it trains on triplets
        return processing_method(row[0], 3), tf.constant(True, dtype=tf.bool)


def data_pipeline(filenames, batch_size, num_epochs=None, min_after_dequeue=10000, allow_smaller_final_batch=False):
    """
    Creates a pipeline for the data contained within the given files.
    It does this using TensorFlow queues.

    @return: A tuple in which the first element is a graph operation
    which gets a random batch of the data, and the second element is
    a graph operation to get a batch of the labels corresponding
    to the data gotten by the first element of the tuple.

    Notes:
    1) Maybe should be using sparse tensors
    """
    with tf.name_scope("data_pipeline"):
        # with tf.device("/cpu:0"):
        capacity = min_after_dequeue + 3 * batch_size
        filename_queue = tf.train.string_input_producer(filenames, capacity=capacity, num_epochs=num_epochs)
        example_op, label_op = acquire_data_ops(filename_queue, full_onehot_process_line_as_2d_input)

        example_batch, label_batch = tf.train.shuffle_batch(
            [example_op, label_op],
            batch_size=batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            num_threads=12,
            allow_smaller_final_batch=allow_smaller_final_batch)

        return example_batch, None


def create_tf_records_input_data_fn_with_ep(filenames, batch_size, shuffle_buffer_size=100000):
    """
    NOTES:
    1) I likely won't be using this, opting to instead use the onehot implementation
    """
    def tf_records_input_data_fn():
        dataset = tf.data.TFRecordDataset(filenames)  # , buffer_size=100000)

        the_values = tf.constant(
            [[0, 0, 0, 0, 0, 0, 0, 0],  # 0
             [0, 0, 0, 0, 0, 0, 1, 0],  # 1
             [1, 0, 0, 0, 0, 0, 0, 0],  # K
             [0, 1, 0, 0, 0, 0, 0, 0],  # Q
             [0, 0, 1, 0, 0, 0, 0, 0],  # R
             [0, 0, 0, 1, 0, 0, 0, 0],  # B
             [0, 0, 0, 0, 1, 0, 0, 0],  # N
             [0, 0, 0, 0, 0, 1, 0, 0],  # P
             [0, 0, 0, 0, 0, 0, 0, 1],  # C
             [-1, 0, 0, 0, 0, 0, 0, 0],  # k
             [0, -1, 0, 0, 0, 0, 0, 0],  # q
             [0, 0, -1, 0, 0, 0, 0, 0],  # r
             [0, 0, 0, -1, 0, 0, 0, 0],  # b
             [0, 0, 0, 0, -1, 0, 0, 0],  # n
             [0, 0, 0, 0, 0, -1, 0, 0],  # p
             [0, 0, 0, 0, 0, 0, 0, -1],  # c
             ], dtype=tf.float32)

        def parser(record):
            keys_to_features = {
                "boards": tf.FixedLenFeature([8 * 8 * 3], tf.int64),  # , default_value = []),
            }

            parsed = tf.parse_single_example(record, keys_to_features)
            boards_tensor = tf.gather(the_values, parsed["boards"])
            # boards_tensor = tf.reshape(squares_tensors, shape=[3,8,8,8])


            return {"boards_tensor": boards_tensor}

        def reshape_fn(batch):
            return tf.reshape(batch["boards_tensor"], [-1, 8, 8, 8])

        dataset = dataset.map(parser, num_parallel_calls=100)
        # dataset = dataset.map(lambda x : tf.gather(the_values, ))
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(reshape_fn, num_parallel_calls=100)
        dataset = dataset.repeat()

        iterator = dataset.make_one_shot_iterator()

        features = iterator.get_next()

        return features, None

    return tf_records_input_data_fn


def one_hot_create_tf_records_input_data_fn(filenames, batch_size, shuffle_buffer_size=100000):
    def tf_records_input_data_fn():
        dataset = tf.data.TFRecordDataset(filenames)  # , buffer_size=100000)

        def parser(record):
            keys_to_features = {
                "boards": tf.FixedLenFeature([8 * 8 * 3], tf.int64),  # , default_value = []),
            }

            return tf.parse_single_example(record, keys_to_features)["boards"]
            # boards_tensor = tf.gather(the_values, parsed["boards"])
            # boards_tensor = tf.reshape(squares_tensors, shape=[3,8,8,8])
            # return  {"boards_tensor" : boards_tensor}

        def reshape_fn(batch):
            # return {"boards_tensor" : tf.reshape(batch, [-1, 8,8,8])}
            print(batch)
            temp_value = tf.reshape(batch, [-1, 8, 8])
            return temp_value

        dataset = dataset.map(parser, num_parallel_calls=100)
        # dataset = dataset.map(lambda x : tf.gather(the_values, ))
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(reshape_fn, num_parallel_calls=100)
        dataset = dataset.map(lambda x: tf.one_hot(x, 16))

        dataset = dataset.repeat()

        iterator = dataset.make_one_shot_iterator()

        features = iterator.get_next()
        return features, None

    return tf_records_input_data_fn


def move_gen_one_hot_create_tf_records_input_data_fn(filenames, batch_size, shuffle_buffer_size=100000, repeat=True, shuffle=True):
    def tf_records_input_data_fn():
        dataset = tf.data.TFRecordDataset(filenames)

        def parser(record):
            context_features = {
                "board": tf.FixedLenFeature([64], tf.int64),
            }

            sequence_features = {
                "moves": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                "move_scores": tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            }


            parsed_record = tf.parse_single_sequence_example(record, context_features=context_features,sequence_features=sequence_features)

            reshaped_board = tf.reshape(parsed_record[0]["board"],[8,8])
            sparse_tensor = tf.sparse_tensor_to_dense(tf.SparseTensor(tf.expand_dims(parsed_record[1]["moves"],1), parsed_record[1]["move_scores"],[1792]),validate_indices=False)

            return reshaped_board, sparse_tensor


        dataset = dataset.map(parser, num_parallel_calls=100)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

        dataset = dataset.batch(batch_size)

        dataset = dataset.map(lambda x,y:(tf.one_hot(x,16),y))

        if repeat:
            dataset = dataset.repeat()

        iterator = dataset.make_one_shot_iterator()

        features = iterator.get_next()

        return features[0],features[1]

    return tf_records_input_data_fn


def input_data_fn(filenames, batch_size, epochs, min_after_dequeue, allow_smaller_final_batch=False):
    """
    A function which is called to create the data pipeline ops made by the function
    data_pipeline.  It does this in a way such that they belong to the session managed
    by the Estimator object that will be given this function.
    """
    with tf.name_scope("input_fn"):
        batch, labels = data_pipeline(
            filenames=filenames,
            batch_size=batch_size,
            num_epochs=epochs,
            min_after_dequeue=min_after_dequeue,
            allow_smaller_final_batch=allow_smaller_final_batch)
        return batch, labels


def serving_input_reciever_fn_creater(whites_turn):
    """
    TO MAKE BLACKS TURN:
    1) switch color of pieces (switch occupied_w with occupied_b
    2) reverse the positioning of the board
    """
    def serving_input_reciever_fn():
        feature_spec = {"occupied_w" : tf.FixedLenFeature([1], tf.int64),
                        "occupied_b" : tf.FixedLenFeature([1], tf.int64),
                        "kings" : tf.FixedLenFeature([1], tf.int64),
                        "queens": tf.FixedLenFeature([1], tf.int64),
                        "rooks": tf.FixedLenFeature([1], tf.int64),
                        "bishops" : tf.FixedLenFeature([1], tf.int64),
                        "knights": tf.FixedLenFeature([1], tf.int64),
                        "pawns": tf.FixedLenFeature([1], tf.int64),
                        "ep_square" : tf.FixedLenFeature([1], tf.int64),
                        "castling_rights" : tf.FixedLenFeature([1], tf.int64),
                        }



        serialized_tf_example = tf.placeholder(dtype=tf.string, name='input_example_tensor')

        receiver_tensors = {'example': serialized_tf_example}

        features = tf.parse_example(serialized_tf_example, feature_spec)

        print("features:", features)
        if whites_turn:
            first_players_occupied = features["occupied_w"]
            second_players_occupied = features["occupied_b"]
        else:
            first_players_occupied = features["occupied_b"]
            second_players_occupied = features["occupied_w"]




        the_ints = tf.concat([
            tf.bitwise.invert(tf.bitwise.bitwise_or(features["occupied_w"], features["occupied_b"])),# (Empty squares) might be faster to just pass occupied in the Protocol Buffer
            features["ep_square"],# (ep_square) very likely should do this differently (to avoid 8 indicies in tf.gather and instead use 1)
            tf.bitwise.bitwise_and(first_players_occupied, features["kings"]),# Likely can do without AND operation
            tf.bitwise.bitwise_and(first_players_occupied, features["queens"]),
            tf.bitwise.bitwise_and(first_players_occupied, features["rooks"]),
            tf.bitwise.bitwise_and(first_players_occupied, features["bishops"]),
            tf.bitwise.bitwise_and(first_players_occupied, features["knights"]),
            tf.bitwise.bitwise_and(first_players_occupied, features["pawns"]),
            tf.bitwise.bitwise_and(first_players_occupied, features["castling_rights"]),# (White castling rooks) very likely should do this differently (to avoid 8 indicies in tf.gather and instead use 1)
            tf.bitwise.bitwise_and(second_players_occupied, features["kings"]),# Likely can do without AND operation
            tf.bitwise.bitwise_and(second_players_occupied, features["queens"]),
            tf.bitwise.bitwise_and(second_players_occupied, features["rooks"]),
            tf.bitwise.bitwise_and(second_players_occupied, features["bishops"]),
            tf.bitwise.bitwise_and(second_players_occupied, features["knights"]),
            tf.bitwise.bitwise_and(second_players_occupied, features["pawns"]),
            tf.bitwise.bitwise_and(second_players_occupied, features["castling_rights"]),# (Black castling rooks) very likely should do this differently (to avoid 8 indicies in tf.gather and instead use 1)
            ],
            axis=1
        )

        print("the_ints:", the_ints)

        the_bytes = tf.cast(tf.bitcast(the_ints,tf.uint8),dtype=tf.int32)

        print("the_bytes:", the_bytes)

        float_bool_masks = tf.constant(
            [np.unpackbits(num).tolist() for num in np.arange(2 ** 8, dtype=np.uint8)],
            dtype=tf.float32)

        print("float_bool_masks",float_bool_masks)

        data = tf.gather(float_bool_masks, the_bytes)

        print("data:", data)

        properly_aranged_data = tf.transpose(data, perm=[0, 2, 3, 1])
        print("properly_aranged_data:", properly_aranged_data)

        if not whites_turn:
            THE_AXIS_TO_REVERSE = 1 ####################################################################################################I AM NOT CONFIDENT ABOUT THIS IT MAY BE 1#####################################################################################
            properly_aranged_data = tf.reverse(properly_aranged_data, axis=[THE_AXIS_TO_REVERSE])

            print("new properly_aranged_data:", properly_aranged_data)

        return tf.estimator.export.ServingInputReceiver(properly_aranged_data, receiver_tensors)

    return serving_input_reciever_fn


def board_as_str_serving_input_receiver_fn():
    """
    A function to use for input processing when serving the model from string representation of the board.
    """
    feature_spec = {'str': tf.FixedLenFeature([1], tf.string)}
    serialized_tf_example = tf.placeholder(dtype=tf.string, name='input_example_tensor')

    receiver_tensors = {'example' : serialized_tf_example}

    features = tf.parse_example(serialized_tf_example, feature_spec)

    features['str'] = tf.reshape(features['str'], [-1])

    data = full_onehot_process_line_as_2d_input(features['str'], for_serving=True)

    return tf.estimator.export.ServingInputReceiver(data, receiver_tensors)


def line_counter(filename):
    """
    A function to count the number of lines in a file (kinda) efficiently.
    """
    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b:
                break
            yield b

    with open(filename, "r") as f:
        return sum(bl.count("\n") for bl in blocks(f))


class ValidationRunHook(tf.train.SessionRunHook):
    """
    A subclass of tf.train.SessionRunHook to be used to evaluate validation data
    efficiently during an Estimator's training run.


    TO DO:
    1) Figure out how to handle steps to do one complete epoch
    2) Have this not call the evaluate function because it has to restore from a
    checkpoint, it will likely be faster if I evaluate it on the current training graph
    3) Implement an epoch counter to be printed along with the validation results
    """

    def __init__(self, step_increment, estimator, input_fn_creator, temp_num_steps_in_epoch=None, recall_input_fn_creator_after_evaluate=False):
        self.step_increment = step_increment
        self.estimator = estimator
        self.input_fn_creator = input_fn_creator
        self.recall_input_fn_creator_after_evaluate = recall_input_fn_creator_after_evaluate
        self.temp_num_steps_in_epoch = temp_num_steps_in_epoch


    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()

        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use ValidationRunHook.")

        self._input_fn = self.input_fn_creator()

    def after_create_session(self, session, coord):
        self._step_started = session.run(self._global_step_tensor)

    def before_run(self, run_context):
        return training.session_run_hook.SessionRunArgs(self._global_step_tensor)


    def after_run(self, run_context, run_values):
        if (run_values.results - self._step_started) % self.step_increment == 0:
            print(self.estimator.evaluate(
                input_fn=self._input_fn,
                steps=self.temp_num_steps_in_epoch,
            ))

            if self.recall_input_fn_creator_after_evaluate:
                self._input_fn =  self.input_fn_creator()

