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


def board_eval_model_fn(features, labels, mode, params):
    """
    Generates an EstimatorSpec for the model.
    """

    if mode == tf.estimator.ModeKeys.PREDICT:
        input_layer = features["feature"]
    else:
        #Reshape features from original shape of [-1, 3, 8, 8, 16]
        input_layer = tf.reshape(features, [-1, 8, 8, params['num_input_filters']])

    cur_inception_module = input_layer

    activation_summaries = []
    for module_num, module_shape in enumerate(params['inception_modules']):
        if callable(module_shape):
            cur_inception_module = module_shape(cur_inception_module)
        else:
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


    # Calculate loss (for both TRAIN and EVAL modes)
    if mode != tf.estimator.ModeKeys.PREDICT:
        to_split = tf.reshape(logits, [-1, 3, 1])
        original_pos, desired_pos, random_pos = tf.split(to_split, [1, 1, 1], 1)


        # Implementing an altered version of the loss function defined in Deep Pink
        # There are a few other methods I've been trying out in commented out, though none seem to be as good as
        # the one proposed in Deep Pink
        with tf.variable_scope("loss"):
            # adjusted_equality_sum = (tf.scalar_mul(1.02,original_pos)+ desired_pos)
            adjusted_equality_sum = (original_pos + desired_pos)


            real_greater_rand_scalar_loss = tf.reduce_mean(-tf.log(tf.sigmoid(desired_pos - random_pos)))

            # to_compare =tf.squeeze(tf.concat([desired_pos, random_pos], 1))
            # to_compare = tf.reshape(to_split, [-1,3])
            # the_labels = tf.squeeze(tf.concat([tf.ones_like(desired_pos),tf.zeros_like(random_pos)], 1))

            # softmax_inequality_loss = tf.losses.sigmoid_cross_entropy(the_labels, to_compare)

            ## test_new_loss_component = tf.reduce_mean(-tf.log(tf.sigmoid(-(original_pos + random_pos))))
            ## test_new_loss_component_summary = tf.summary.scalar("test_new_loss_component", test_new_loss_component)

            # old_new_squared_scalar_loss = tf.reduce_mean(tf.square(adjusted_equality_sum))
            # loss = real_greater_rand_scalar_loss +  old_new_squared_scalar_loss

            ## test_equality_loss = tf.reduce_mean(
            #     -tf.log(tf.sigmoid(adjusted_equality_sum)) - tf.log(tf.sigmoid( -adjusted_equality_sum)))
            equality_scalar_loss = tf.reduce_mean(-tf.log(tf.sigmoid(adjusted_equality_sum)))
            negative_equality_scalar_loss = tf.reduce_mean(-tf.log(tf.sigmoid( -adjusted_equality_sum)))

            ratio_old_new_sum_loss_to_negative_sum = tf.divide(equality_scalar_loss, negative_equality_scalar_loss)

            real_rand_loss_summary = tf.summary.scalar("real_greater_rand_loss", real_greater_rand_scalar_loss)


            equality_sum_loss_summary = tf.summary.scalar("mean_original_plus_desired_loss", equality_scalar_loss)
            negative_equality_sum_loss_summary = tf.summary.scalar("mean_negative_original_plus_desired", negative_equality_scalar_loss)


            # loss = real_greater_rand_scalar_loss + test_equality_loss
            loss = real_greater_rand_scalar_loss + equality_scalar_loss + negative_equality_scalar_loss
            # loss = real_greater_rand_scalar_loss + equality_scalar_loss + negative_equality_scalar_loss + test_new_loss_component

            # loss = old_new_squared_scalar_loss + softmax_inequality_loss
            # loss = softmax_inequality_loss

            # old_real_summary = tf.summary.scalar("old_real_squared_loss", old_new_squared_scalar_loss)
            # softmax_summary = tf.summary.scalar("softmax_inequality_loss", softmax_inequality_loss)  #THIS SHOULD ALL ACTUALLY BE CALLED SIGMOID INEQUALITY LOSS@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            loss_summary = tf.summary.scalar("loss", loss)


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
        # "real_rand_guessing": tf.greater(desired_pos, random_pos)
        "scores" : logits
    }

    # A dictionary for scoring used when exporting model for serving.
    the_export_outputs = {
        "serving_default" : tf.estimator.export.RegressionOutput(value=logits)}

    unstacked_input = tf.unstack(input_layer, axis=3)
    # Create the validation metrics
    validation_metric = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        real_rand_diff = desired_pos - random_pos
        old_plus_desired = original_pos + desired_pos

        mean_abs_real_rand_diff = tf.reduce_mean(tf.abs(real_rand_diff))
        mean_abs_old_plus_desired = tf.reduce_mean(tf.abs(old_plus_desired))

        abs_realrand_realold_ratio = tf.reduce_mean(real_rand_diff) / mean_abs_old_plus_desired#################################################################################RENAME THIS AND IT'S COMMENT##############################################################################################################


        validation_metric = {
            "metrics/rand_vs_real_accuracy": mean_metric_creator("metrics/rand_vs_real_accuracy")(tf.cast(tf.greater(desired_pos, random_pos), tf.float32), None),
            "metrics/mean_dist_real_rand": mean_metric_creator("metrics/mean_dist_real_rand")(real_rand_diff, None),
            # "metrics/mean_abs_dist_real_rand": mean_metric_creator("metrics/mean_abs_dist_real_rand")(abs_real_rand_diff, None),
            "metrics/mean_abs_dist_real_rand": (mean_abs_real_rand_diff, tf.summary.scalar("metrics/mean_abs_dist_real_rand", mean_abs_real_rand_diff)),
            "metrics/mean_dist_old_real": mean_metric_creator("metrics/mean_dist_old_real")(old_plus_desired, None),
            "metrics/mean_abs_dist_old_real": (mean_abs_old_plus_desired, tf.summary.scalar("metrics/mean_abs_dist_old_real", mean_abs_old_plus_desired)),
            "metrics/abs_realrand_realold_ratio" : (abs_realrand_realold_ratio, tf.summary.scalar("metrics/abs_realrand_realold_ratio", abs_realrand_realold_ratio)),

            # "loss/test_new_loss_component" : (test_new_loss_component,test_new_loss_component_summary),
            # "loss/real_greater_rand_loss": (real_greater_rand_scalar_loss, real_rand_loss_summary),
            "loss/mean_original_plus_desired_loss": (equality_scalar_loss, equality_sum_loss_summary),
            "loss/mean_negative_original_plus_desired": (negative_equality_scalar_loss, negative_equality_sum_loss_summary),
            # "loss/old_real_squared_loss" : (old_new_squared_scalar_loss, old_real_summary),
            # "loss/softmax_inequality_loss":(softmax_inequality_loss, softmax_summary),
            "loss/loss" : (loss, loss_summary),
            # "loss/ratio_old_new_sum_loss_to_negative_sum" : (ratio_old_new_sum_loss_to_negative_sum, tf.summary.scalar("loss/ratio_old_new_sum_loss_to_negative_sum", ratio_old_new_sum_loss_to_negative_sum)),
            # "sanity_check/input_tensor_mean" : mean_metric_creator("sanity_check/input_tensor_mean")(features, None),
            "metrics/mean_old_pos": mean_metric_creator("metrics/mean_old_pos")(original_pos, None),
            "metrics/mean_new_pos": mean_metric_creator("metrics/mean_new_pos")(desired_pos, None),
            # "metrics/negative_new_to_old_ratio" : mean_metric_creator("metrics/negative_new_to_old_ratio")(tf.divide(-desired_pos, original_pos), None),
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
        print(features)
        input_layer = features["data"]

        # legal_move_indices = features["move_indices"]
    else:
        input_layer = features


    cur_inception_module = input_layer

    activation_summaries = []
    for module_num, module_shape in enumerate(params['inception_modules']):
        if callable(module_shape):
            cur_inception_module = module_shape(cur_inception_module)
        else:
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

    if not params["conv_init_fn"] is None:
        params["conv_init_fn"]()

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
        #THE USE OF tf.scatter_nd FUNCTION WOULD LIKELY SIMPLIFY ALL OF THIS

        weight_mask = labels

        bool_mask = tf.cast(weight_mask, tf.bool)
        weight_mask = tf.to_float(bool_mask)

        opposite_bool_mask = tf.logical_not(bool_mask)
        important_logits = tf.boolean_mask(logits, bool_mask)
        important_labels = tf.boolean_mask(labels, bool_mask)


        total_possible_moves = tf.reduce_sum(weight_mask)
        # mean_possible_moves = tf.reduce_mean(tf.reduce_sum(total_possible_moves, axis=1))



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
            loss_scalar_summary = tf.summary.scalar("loss", loss)



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
        "move_values": logits,}
        # "important_logits":important_logits}


    if mode != tf.estimator.ModeKeys.PREDICT:
        # A dictionary for scoring used when exporting model for serving.
        the_export_outputs = {
            # "serving_default" : tf.estimator.export.PredictOutput(outputs={"logits": logits}),
            "serving_default" : tf.estimator.export.ClassificationOutput(scores=logits),
        }
        # A dictionary for scoring used when exporting model for serving.
    else:
        # real_indices = index_tensor_to_index_pairs(legal_move_indices)
        # legal_move_logits = tf.gather_nd(logits, real_indices)
        the_export_outputs = {
            # "serving_default" : tf.estimator.export.PredictOutput(outputs={"logits": logits}),
            "serving_default": tf.estimator.export.ClassificationOutput(scores=logits),
            # "legal_moves" :  tf.estimator.export.ClassificationOutput(scores=legal_move_logits),
        }


    unstacked_input = tf.unstack(input_layer, axis=3)
    # Create the validation metrics
    validation_metric = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        validation_metric = {
            "loss/loss" : (loss, loss_scalar_summary),
            "metrics/total_moves_above_desired_moves" : (total_moves_above_desired_moves, tf.summary.scalar("metrics/total_moves_above_desired_moves", total_moves_above_desired_moves)),
            "metrics/ratio_moves_above_desired_moves": (total_moves_above_desired_moves/total_possible_moves,tf.summary.scalar("metrics/ratio_moves_above_desired_moves",total_moves_above_desired_moves/total_possible_moves)),
            # "sanity_check/mean_possible_moves" : (mean_possible_moves, tf.summary.scalar("sanity_check/mean_possible_moves", mean_possible_moves)),
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




def one_hot_create_tf_records_input_data_fn(filename_pattern, batch_size, include_unoccupied=True, shuffle_buffer_size=None):################MUST COMBINE THIS FUNCTIONALITY WITH THE ONE BELOW IT (just add ability to subtract one before one_hot
    def tf_records_input_data_fn():
        with tf.device('/cpu:0'):

            filenames = tf.data.Dataset.list_files(filename_pattern)
            dataset = filenames.apply(
                tf.contrib.data.parallel_interleave(
                    lambda filename : tf.data.TFRecordDataset(filename),
                    cycle_length=7,
                    sloppy=True))

            def parser(record):
                keys_to_features = {
                    "boards": tf.FixedLenFeature([8 * 8 * 3], tf.int64)}  # , default_value = []),


                return tf.reshape(tf.parse_single_example(record, keys_to_features)["boards"], [-1, 8,8])

            if not shuffle_buffer_size is None:
                dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)


            num_things_in_parallel = 12
            dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parser, batch_size=batch_size, num_parallel_batches=num_things_in_parallel))

            if include_unoccupied:
                dataset = dataset.map(lambda x: tf.one_hot(x, 16), num_parallel_calls=num_things_in_parallel)
            else:
                dataset = dataset.map(lambda x: tf.one_hot(x-1, 15), num_parallel_calls=num_things_in_parallel)

            dataset = dataset.prefetch(1)#num_things_in_parallel)#buffer_size=batch_size)

            dataset = dataset.repeat()

            iterator = dataset.make_one_shot_iterator()

            features = iterator.get_next()
            return features, None

    return tf_records_input_data_fn


def move_gen_one_hot_create_tf_records_input_data_fn(filename_pattern, batch_size, shuffle_buffer_size=100000, include_unoccupied=True, repeat=True, shuffle=True):
    def tf_records_input_data_fn():

        filenames = tf.data.Dataset.list_files(filename_pattern)
        dataset = filenames.apply(
            tf.contrib.data.parallel_interleave(
                lambda filename: tf.data.TFRecordDataset(filename),
                cycle_length=7,
                sloppy=True,
            ))

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


        dataset = dataset.map(parser, num_parallel_calls=12)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

        dataset = dataset.batch(batch_size)

        if include_unoccupied:
            dataset = dataset.map(lambda x,y:(tf.one_hot(x,16),y))
        else:
            dataset = dataset.map(lambda x,y:(tf.one_hot(x-1,15),y))

        dataset = dataset.prefetch(1)

        if repeat:
            dataset = dataset.repeat()

        iterator = dataset.make_one_shot_iterator()

        features = iterator.get_next()

        return features[0],features[1]

    return tf_records_input_data_fn



def serving_input_reciever_fn_creater(whites_turn):#########################################################################DECIDE IF THIS SHOULD BE REMOVED FOR THIS COMMIT###################################################
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

        with tf.device("/gpu:0"):
            print("features:", features)
            if whites_turn:
                first_players_occupied = features["occupied_w"]
                second_players_occupied = features["occupied_b"]
            else:
                first_players_occupied = features["occupied_b"]
                second_players_occupied = features["occupied_w"]


            the_ints = tf.concat([
                tf.bitwise.invert(tf.bitwise.bitwise_or(features["occupied_w"], features["occupied_b"])),# (Empty squares) might be faster to just pass occupied in the Protocol Buffer
                features["ep_square"],# (ep_square) very likely should do this differently (to avoid 8 indicies in tf.gather and instead use 1)#########################################################THIS I THINK IS COMPLETELY WRONG SINCE I NEVER DID THE GATHER###########################
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

            the_bytes = tf.cast(tf.bitcast(the_ints,tf.uint8),dtype=tf.int32)

            if not whites_turn:
                the_bytes = tf.reverse(the_bytes, axis=[2])

            float_bool_masks = tf.constant(
                [np.unpackbits(num).tolist() for num in np.arange(2 ** 8, dtype=np.uint8)],
                dtype=tf.float32)

            data = tf.gather(float_bool_masks, the_bytes)

            properly_aranged_data = tf.transpose(data, perm=[0, 2, 3, 1])

            return tf.estimator.export.ServingInputReceiver(properly_aranged_data, receiver_tensors)

    return serving_input_reciever_fn


def serving_input_reciever_legal_moves_fn(whites_turn):
    """
    TO MAKE BLACKS TURN:
    1) switch color of pieces (switch occupied_w with occupied_b)
    2) reverse the positioning of the board
    3) (not done yet) switch the indices of logits to return
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
                        "legal_move_indices" : tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                        }



        serialized_tf_example = tf.placeholder(dtype=tf.string, name='input_example_tensor')

        receiver_tensors = {'example': serialized_tf_example}

        features = tf.parse_example(serialized_tf_example, feature_spec)

        with tf.device("/gpu:0"):
            if whites_turn:
                first_players_occupied = features["occupied_w"]
                second_players_occupied = features["occupied_b"]

                the_legal_move_indices = features["legal_move_indices"] #currently this has no use
            else:
                first_players_occupied = features["occupied_b"]
                second_players_occupied = features["occupied_w"]

                the_legal_move_indices = features["legal_move_indices"] #currently this has no use


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

            the_bytes = tf.cast(tf.bitcast(the_ints,tf.uint8),dtype=tf.int32)

            if not whites_turn:
                the_bytes = tf.reverse(the_bytes, axis=[2])


            float_bool_masks = tf.constant(
                [np.unpackbits(num).tolist() for num in np.arange(2 ** 8, dtype=np.uint8)],
                dtype=tf.float32)

            data = tf.gather(float_bool_masks, the_bytes)

            properly_aranged_data = tf.transpose(data, perm=[0, 2, 3, 1])

            dict_to_return = {"data": properly_aranged_data, "move_indices": the_legal_move_indices}
            return tf.estimator.export.ServingInputReceiver(dict_to_return, receiver_tensors)

    return serving_input_reciever_fn


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
        if (run_values.results - self._step_started) % self.step_increment == 0 and run_values.results != 0:
            print(self.estimator.evaluate(
                input_fn=self._input_fn,
                steps=self.temp_num_steps_in_epoch,
            ))

            if self.recall_input_fn_creator_after_evaluate:
                self._input_fn =  self.input_fn_creator()

