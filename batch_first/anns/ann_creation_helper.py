import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from tensorflow.python import training

from functools import reduce

from ..chestimator import new_get_board_data
# from ..board_jitclass import generate_move_to_enumeration_dict


#If you don't have TensorRT installed, you can just comment out it's use. Here it's only used in the
#save_model_as_graph_def_for_serving function.
from tensorflow.contrib import tensorrt as trt





def save_model_as_graphdef_for_serving(model_path, output_model_path, output_filename, output_node_name, model_tags="serve", as_text=False):
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.3))) as sess:
        board_eval_graph = tf.train.import_meta_graph(tf.saved_model.loader.load(sess, [model_tags], model_path))

        constant_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), [output_node_name])

        trt_graph = trt.create_inference_graph(constant_graph_def,
                                   [output_node_name],
                                   max_batch_size=25000,
                                               precision_mode="FP32",
                                               max_workspace_size_bytes=5000000000
                                   )


        tf.train.write_graph(trt_graph, output_model_path, output_filename, as_text=as_text)



def build_fully_connected_layers_with_batch_norm(the_input, shape, kernel_initializer, mode, num_previous_fully_connected_layers=0, activation_summaries=[], scope_prefix=""):
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
                kernel_initializer=kernel_initializer(),
                name="layer")

            temp_batch_normalized = tf.layers.batch_normalization(temp_pre_activation,
                                                                  training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                                  fused=True)

            temp_layer_output = tf.nn.relu(temp_batch_normalized)

            the_input = temp_layer_output

        activation_summaries.append(layers.summarize_activation(temp_layer_output))

    return the_input, activation_summaries


def build_inception_module_with_batch_norm(the_input, module, kernel_initializer, mode, activation_summaries=[], num_previously_built_inception_modules=0, padding='same', force_no_concat=False,make_trainable=True, weight_regularizer=None):
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
    if weight_regularizer is None:
        weight_regularizer = lambda:None

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
                    kernel_initializer=kernel_initializer(),
                    kernel_regularizer=weight_regularizer(),
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
        if len(path_outputs) == 1:
            return path_outputs[0], activation_summaries
        for j in range(1, len(path_outputs)):
            if force_no_concat or path_outputs[0].get_shape().as_list()[1:3] != path_outputs[j].get_shape().as_list()[1:3]:
                return [temp_input for temp_input in path_outputs], activation_summaries

        return tf.concat([temp_input for temp_input in path_outputs], 3), activation_summaries
    
    
    

def metric_dict_creator(the_dict):
    metric_dict = {}
    for key, value in the_dict.items():
        if isinstance(value, tuple): # Given a tuple (tensor, summary)
            metric_dict[key] = (tf.reduce_mean(value[0]), value[1])
        else: #Given a tensor
            mean_value = tf.reduce_mean(value)
            metric_dict[key] = (mean_value, tf.summary.scalar(key, mean_value))

    return metric_dict
    
    



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
                params['kernel_initializer'],
                mode,
                padding='valid',
                make_trainable=params['trainable_cnn_modules'],
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

    print(tf.train.get_global_step)
    if not params["conv_init_fn"] is None: #and tf.train.global_step(tf.get_default_session(), tf.train.get_global_step()) == 0:
        params["conv_init_fn"]()


    # Build the fully connected layers
    dense_layers_outputs, activation_summaries = build_fully_connected_layers_with_batch_norm(
        inception_module_outputs,
        params['dense_shape'],
        params['kernel_initializer'],
        mode,
        activation_summaries=activation_summaries)


    # Create the final layer of the ANN
    logits = tf.layers.dense(inputs=dense_layers_outputs,
                             units=params['num_outputs'],
                             use_bias=False,
                             activation=None,
                             kernel_initializer=params['kernel_initializer'](),
                             name="logit_layer")

    loss = None
    train_op = None
    ratio_old_new_sum_loss_to_negative_sum = None


    # Calculate loss (for both TRAIN and EVAL modes)
    if mode != tf.estimator.ModeKeys.PREDICT:
        to_split = tf.reshape(logits, [-1, 3])
        original_pos, desired_pos, random_pos = tf.split(to_split, [1, 1, 1], 1)


        # Implementing an altered version of the loss function defined in Deep Pink
        # There are a few other methods I've been trying out in commented out, though none seem to be as good as
        # the one proposed in Deep Pink
        with tf.variable_scope("loss"):
            ######################################################################################################
            # adjusted_equality_sum = (original_pos + CONSTANT + desired_pos)
            adjusted_equality_sum = 2*(original_pos + desired_pos)
            adjusted_real_rand_sum = (random_pos - desired_pos)


            # real_greater_rand_scalar_loss = tf.reduce_mean(-tf.log(tf.sigmoid(desired_pos - random_pos)))
            real_greater_rand_scalar_loss = tf.reduce_mean(-tf.log(tf.sigmoid(adjusted_real_rand_sum)))





            ## test_new_loss_component = tf.reduce_mean(-tf.log(tf.sigmoid(-(original_pos + random_pos))))
            ## test_new_loss_component_summary = tf.summary.scalar("test_new_loss_component", test_new_loss_component)


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
            loss_summary = tf.summary.scalar("loss", loss)

            ########################################################################################################

            # # to_compare =tf.squeeze(tf.concat([desired_pos, random_pos], 1))
            # to_compare = tf.squeeze(tf.concat([random_pos,desired_pos], 1))
            # # to_compare = tf.reshape(to_split, [-1,3])
            # the_labels = tf.squeeze(tf.concat([tf.ones_like(desired_pos),tf.zeros_like(random_pos)], 1))
            #
            # old_new_squared_scalar_loss = tf.reduce_mean(tf.square((original_pos + desired_pos)))
            # softmax_inequality_loss = tf.losses.sigmoid_cross_entropy(the_labels, to_compare)
            #
            # loss = old_new_squared_scalar_loss + softmax_inequality_loss
            # # loss = softmax_inequality_loss
            #
            # old_real_summary = tf.summary.scalar("old_real_squared_loss", old_new_squared_scalar_loss)
            # softmax_summary = tf.summary.scalar("softmax_inequality_loss", softmax_inequality_loss)  #THIS SHOULD ALL ACTUALLY BE CALLED SIGMOID INEQUALITY LOSS@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # loss_summary = tf.summary.scalar("loss", loss)


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



    # Create the validation metrics
    validation_metrics = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        # unstacked_input = tf.unstack(input_layer, axis=3)


        rand_real_diff = random_pos - desired_pos
        old_plus_desired = original_pos + desired_pos

        abs_rand_real_diff = tf.abs(rand_real_diff)
        # mean_abs_rand_real_diff = tf.reduce_mean(abs_rand_real_diff)

        abs_old_plus_desired = tf.abs(old_plus_desired)
        mean_abs_old_plus_desired = tf.reduce_mean(abs_old_plus_desired)

        abs_randreal_realold_ratio = tf.reduce_mean(rand_real_diff) / mean_abs_old_plus_desired



        to_create_metric_dict = {
            "metrics/rand_vs_real_accuracy" : tf.cast(tf.less(desired_pos, random_pos), tf.float32),
            "metrics/mean_dist_rand_real" : rand_real_diff,
            "metrics/mean_abs_rand_real_diff" : abs_rand_real_diff,
            "metrics/mean_dist_old_real" : old_plus_desired,
            "metrics/mean_abs_dist_old_real" : abs_old_plus_desired,
            "metrics/abs_randreal_realold_ratio" : abs_randreal_realold_ratio,

            "metrics/mean_old_pos" : original_pos,
            "metrics/mean_new_pos": desired_pos,
            "metrics/mean_random_pos": random_pos,
            "metrics/mean_abs_old_pos": tf.abs(original_pos),
            "metrics/mean_abs_new_pos": tf.abs(desired_pos),
            "metrics/mean_abs_random_pos": tf.abs(random_pos),

            "loss/real_greater_rand_loss" : (real_greater_rand_scalar_loss, real_rand_loss_summary),
            "loss/mean_original_plus_desired_loss" : (equality_scalar_loss, equality_sum_loss_summary),
            "loss/mean_negative_original_plus_desired" : (negative_equality_scalar_loss, negative_equality_sum_loss_summary),
            "loss/loss" : (loss, loss_summary),
            "loss/ratio_old_new_sum_loss_to_negative_sum" : ratio_old_new_sum_loss_to_negative_sum,

        }


        validation_metrics = metric_dict_creator(to_create_metric_dict)

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
        eval_metric_ops=validation_metrics)



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
                params['kernel_initializer'],
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
        params['kernel_initializer'],
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
    validation_metrics = None
    if mode != tf.estimator.ModeKeys.PREDICT:

        mean_important_logits = tf.reduce_mean(important_logits)
        important_logits_minus_labels = important_logits-important_labels
        mean_important_minus_labels = tf.reduce_mean(important_logits_minus_labels)

        to_create_metric_dict = {
            "loss/loss" : (loss, loss_scalar_summary),
            "metrics/total_moves_above_desired_moves" : total_moves_above_desired_moves,
            "metrics/ratio_moves_above_desired_moves" : total_moves_above_desired_moves/total_possible_moves,
            "metrics/mean_evaluation_value" : mean_important_logits ,
            "metrics/mean_expected_value" : important_labels,
            "metrics/mean_abs_expected_value" : abs(important_labels),
            "metrics/distance_from_desired" : mean_important_minus_labels,
            "metrics/abs_distance_from_desired" : tf.abs(important_logits_minus_labels),
            "metrics/distance_from_not_desired" : tf.boolean_mask(logits, opposite_bool_mask) - tf.boolean_mask(labels, opposite_bool_mask),
            "metrics/relative_distance_from_desired": tf.abs(mean_important_minus_labels / mean_important_logits),
        }

        validation_metrics = metric_dict_creator(to_create_metric_dict)


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
        eval_metric_ops=validation_metrics)



def one_hot_create_tf_records_input_data_fn(filename_pattern, batch_size, include_unoccupied=True, shuffle_buffer_size=None): ################MUST COMBINE THIS FUNCTIONALITY WITH THE ONE BELOW IT (just add ability to subtract one before one_hot
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



def no_chestimator_serving_input_reciever():
    piece_bbs, color_occupied_bbs, ep_squares, formatted_data = new_get_board_data()

    receiver_tensors = {"piece_bbs": piece_bbs,
                        "color_occupied_bbs": color_occupied_bbs,
                        "ep_squares": ep_squares}

    return tf.estimator.export.ServingInputReceiver(formatted_data, receiver_tensors)


#The code commented out below has never been run, and as is I'm fairly confident it won't work.

# def no_chestimator_serving_move_scoring_input_reciever(max_batch_size=50000, max_moves_for_a_board=100):
#     piece_bbs, color_occupied_bbs, ep_squares, formatted_data = new_get_board_data()
#
#     moves_per_board = tf.placeholder(tf.uint8, shape=[None], name="moves_per_board_placeholder")
#     moves = tf.placeholder(tf.uint8, shape=[None, 2], name="move_placeholder")
#
#     board_index_repeated_array = tf.transpose(
#         tf.reshape(
#             tf.tile(
#                 tf.range(max_moves_for_a_board),
#                 [max_batch_size]),
#             [max_batch_size, max_moves_for_a_board]),
#         [1, 0])
#
#     move_to_index_array = np.zeros(shape=[64, 64], dtype=np.int32)
#     for key, value in generate_move_to_enumeration_dict().items():
#         move_to_index_array[key[0], key[1]] = value
#
#     move_to_index_tensor = tf.constant(move_to_index_array, shape=[64, 64])
#
#     board_indices_for_moves = tf.boolean_mask(board_index_repeated_array,
#                                               tf.sequence_mask(tf.cast(moves_per_board, tf.int32)))
#
#     move_nums = tf.gather_nd(move_to_index_tensor, tf.cast(moves, tf.int32))
#
#     the_moves = tf.stack([board_indices_for_moves, move_nums], axis=-1)
#
#
#     receiver_tensors = {"piece_bbs" : piece_bbs,
#                         "color_occupied_bbs" : color_occupied_bbs,
#                         "ep_squares" : ep_squares,
#                         "moves_per_board" : moves_per_board,
#                         "moves" : moves}
#
#     dict_for_model_fn = {"data" : formatted_data,
#                          "legal_move_indices" : the_moves}
#
#     return tf.estimator.export.ServingInputReceiver(dict_for_model_fn , receiver_tensors)




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

