import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python import training

from functools import reduce

import batch_first as bf
from batch_first.chestimator import get_board_data


#If you don't have TensorRT installed, you can just comment out it's use. Here it's only used in the
#save_model_as_graph_def_for_serving function.
from tensorflow.contrib import tensorrt as trt



def save_model_as_graphdef_for_serving(model_path, output_model_path, output_filename, output_node_name,
                                       model_tags="serve", trt_memory_fraction=.4, total_video_memory=1.1e10,
                                       max_batch_size=25000, as_text=False):


    #This would ideally be 1 instead of .75, but the GPU that this is running on is responsible for things like graphics
    with tf.Session(config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.75 - trt_memory_fraction))) as sess:

        tf.saved_model.loader.load(sess, [model_tags], model_path)

        constant_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            [output_node_name])

        trt_graph = trt.create_inference_graph(
            constant_graph_def,
            [output_node_name],
            max_batch_size=max_batch_size,
            precision_mode="FP32",
            max_workspace_size_bytes=int(trt_memory_fraction*total_video_memory))

        tf.train.write_graph(trt_graph, output_model_path, output_filename, as_text=as_text)


def build_fully_connected_layers_with_batch_norm(the_input, shape, kernel_initializer, mode,
                                                       activation_summaries=[], scope_prefix=""):
    """
    Builds fully connected layers with batch normalization onto the computational graph of the desired shape.


    The following are a few examples of the shapes that can be given, and how the layers are built

    shape = [512, 256, 128]
    result: input --> 512 --> 256 --> 128

    shape = [[25, 75], [200], 100, 75]
    result: concat((input --> 25 --> 75), (input --> 200)) --> 100 --> 75

    More complicated shapes can be used by having recursive modules like the following
    shape = [[[100], [50, 50]], [200, 50], [75], 100, 20]
    """
    if len(shape) == 0:
        return the_input

    module_outputs = []
    for j, inner_modules in enumerate(shape):
        if isinstance(inner_modules, list):
            output, activation_summaries = build_fully_connected_layers_with_batch_norm(
                the_input,
                inner_modules,
                kernel_initializer,
                mode,
                activation_summaries,
                scope_prefix="%sfc_module_%d/" % (scope_prefix, j + 1),
            )
            module_outputs.append(output)
        else:
            if len(module_outputs) == 1:
                the_input = module_outputs[0]
            elif len(module_outputs) > 1:
                the_input = tf.concat(module_outputs, axis=1)

            for i, layer_shape in enumerate(shape[j:]):

                with tf.variable_scope(scope_prefix + "FC_" + str(i + 1)):
                    pre_activation = tf.layers.dense(
                        inputs=the_input,
                        units=layer_shape,
                        use_bias=False,
                        kernel_initializer=kernel_initializer(),
                        name="layer")

                    batch_normalized = tf.layers.batch_normalization(pre_activation,
                                                                     training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                                     fused=True)

                    the_input = tf.nn.relu(batch_normalized)
                activation_summaries.append(layers.summarize_activation(the_input))

            module_outputs = [the_input]
            break

    if len(module_outputs) == 1:
        return module_outputs[0], activation_summaries

    return tf.concat(module_outputs, axis=1), activation_summaries


def build_inception_module_with_batch_norm(the_input, module, kernel_initializer, mode, activation_summaries=[],
                                           num_previously_built_inception_modules=0, force_no_concat=False,
                                           make_trainable=True, weight_regularizer=None):
    """
    Builds a convolutional module based on a given design.  It returns the final layer/layers in the module,
    and the activation summaries generated for the layers within the inception module.


    The following are a few examples of what can be used in the 'module' parameter.

    module_1 = [[[35,1], (1024, 8)]]

    module_2 = [[[25, 1]],
                [[20,1], [25, 3]],
                [[20,1], [25, 1, 8]],
                [[20,1], [25, 8, 1]]]

    :param module: A list (representing the module's shape), of lists (each representing the shape of a 'path' from the
    input to the output of the module), of either tuples or lists of size 2 or 3 (representing individual layers).
    If a tuple is used, it indicates that the layer should use 'valid' padding, and if a list is used it will use a
    padding of 'same'.  The contents of the list or tuple will be the number of filters to create, followed by either
    one number to indicate the length and width of the filter, or two numbers to indicate them independently if they
    are not equal.
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
                    padding='valid' if isinstance(section, tuple) else 'same',
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


def build_transposed_inception_module_with_batch_norm(the_input, module, kernel_initializer, mode,
                                                      activation_summaries=[],
                                                      num_previously_built_inception_modules=0,
                                                      padding='same', force_no_concat=False, make_trainable=True,
                                                      weight_regularizer=None):
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

                cur_conv_output = tf.layers.conv2d_transpose(
                    inputs=cur_input,
                    filters=section[0],
                    kernel_size=section[1],
                    strides=section[2],
                    padding=padding,
                    use_bias=False,
                    activation=None,
                    kernel_initializer=kernel_initializer(),
                    kernel_regularizer=weight_regularizer(),
                    trainable=make_trainable)
                    # name="layer_" + str(i + 1))

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



def build_convolutional_modules(input, modules, mode, kernel_initializer, kernel_regularizer, make_trainable):
    """
    Creates a desired set of convolutional modules.  Primarily the modules are created through the
    build_inception_module_with_batch_norm function, but if it's functionality proves insufficient, a function can be
    given in place of a modules shape, which will accept the previous modules output as input, and who's output will
    be given to the next module for input. (A detailed example is given below, and more information about the
    shape of a module can be found in the comment for the build_inception_module_with_batch_norm function)



    If the below example of the 'modules' parameter were given, the input Tensor would be given to
    2x2, 3x3, 1x8 and 8x1 convolutional layers (each using a 'same' padding), and then each having their outputs
    concatenated together.

    The concatenated first module output will then be given as input to four 1x1 convolutional layers, three of which
    feed into another convolutional layer (with either filter size 3x3, 1x8, or 8x1), and the output of those
    three layers and the output of the other 1x1 layer are concatenated together (would have shape [-1, 8, 8, 100]).

    This is followed by a 1x1 convolutional layer, and lastly by an 8x8 convolutional layer using a
    'valid' padding, and resulting in a Tensor of shape [-1, 1, 1, 1024], which is reshaped and then returned.


    tf.Shape(input) = [-1, 8, 8, 15]  #Shape of the input given with the following 'modules'
    modules = [
        [[[15, 2]],
         [[25, 3]],
         [[25, 1, 8]],
         [[25, 8, 1]]],

         [[[25, 1]],
         [[20,1], [25, 3]],
         [[20,1], [25, 1, 8]],
         [[20,1], [25, 8, 1]]],

         [[[25, 1], (1024,8)]],

         lambda x: tf.reshape(x, [-1, 1024]),
        ]


    :param modules: A list which sequentially represents the graph operations to be created.  It may contain
     any combination of either functions which accept the previous module's output and return the desired input for
     the next module, or shapes of modules which can be given to the build_inception_module_with_batch_norm method
    """
    activation_summaries = []

    cur_inception_module = input

    for module_num, module_shape in enumerate(modules):
        if callable(module_shape):
            cur_inception_module = module_shape(cur_inception_module)
        else:
            cur_inception_module, activation_summaries = build_inception_module_with_batch_norm(
                cur_inception_module,
                module_shape,
                kernel_initializer,
                mode,
                make_trainable=make_trainable,
                num_previously_built_inception_modules=module_num,
                weight_regularizer=kernel_regularizer)

    if isinstance(cur_inception_module, list):
        inception_module_paths_flattened = [
            tf.reshape(
                path,
                [-1, reduce(lambda a, b: a * b, path.get_shape().as_list()[1:])]
            ) for path in cur_inception_module]
        return tf.concat(inception_module_paths_flattened, 1), activation_summaries

    return cur_inception_module, activation_summaries


def metric_dict_creator(the_dict):
    metric_dict = {}
    for key, value in the_dict.items():
        if isinstance(value, tuple): # Given a tuple (tensor, summary)
            metric_dict[key] = (tf.reduce_mean(value[0]), value[1])
        else: #Given a tensor
            mean_value = tf.reduce_mean(value)
            metric_dict[key] = (mean_value, tf.summary.scalar(key, mean_value))

    return metric_dict


def score_comparison_model_fn(features, labels, mode, params):
    """
    Generates an EstimatorSpec for the model.
    """

    if mode == tf.estimator.ModeKeys.PREDICT:
        input_layer = features['board']
    else:
        # Reshape features from original shape of [-1, 2, 8, 8, params['num_input_filters']]
        input_layer = tf.reshape(features, [-1, 8, 8, params['num_input_filters']])


    inception_module_outputs, activation_summaries = build_convolutional_modules(
        input_layer,
        params['inception_modules'],
        mode,
        params['kernel_initializer'],
        params['kernel_regularizer'],
        params['trainable_cnn_modules'])


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


    # Calculate loss (for both TRAIN and EVAL modes)
    if mode != tf.estimator.ModeKeys.PREDICT:

        reshaped_logits = tf.reshape(logits, [-1, 2])

        first_boards_logits, second_boards_logits = tf.split(reshaped_logits, 2, axis=1)
        first_boards_labels, second_boards_labels = tf.split(labels, 2, axis=1)

        first_board_intended_greater = tf.greater(first_boards_labels, second_boards_labels)

        intended_greater_logits = tf.where(first_board_intended_greater, first_boards_logits, second_boards_logits)
        intended_less_logits = tf.where(first_board_intended_greater, second_boards_logits, first_boards_logits)

        intended_greater_logits_minus_less = intended_greater_logits - intended_less_logits


        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(-tf.log(tf.sigmoid(intended_greater_logits_minus_less)))
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


    # Generate predictions
    predictions = {"scores" : logits}

    # A dictionary for scoring used when exporting model for serving.
    the_export_outputs = {"serving_default" : tf.estimator.export.RegressionOutput(value=logits)}


    # Create the validation metrics
    validation_metrics = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        ratio_better_move_chosen = tf.reduce_mean(tf.cast(tf.greater(intended_greater_logits, intended_less_logits), dtype=tf.float32))

        to_create_metric_dict = {
            "loss/loss": (loss, loss_summary),
            "metrics/mean_evaluation_value" : logits,
            "metrics/mean_abs_evaluation_value": tf.abs(logits),
            "metrics/mean_dist_between_better_worse_boards" : intended_greater_logits_minus_less,
            "metrics/ratio_better_move_chosen" : ratio_better_move_chosen,
        }

        validation_metrics = metric_dict_creator(to_create_metric_dict)

    # Create the trainable variable summaries and merge them together to give to a hook
    trainable_var_summaries = layers.summarize_tensors(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))  # Not sure if needs to be stored as a variable, should check
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


def sf_score_replicator_model_fn(features, labels, mode, params):
    """
    Generates an EstimatorSpec for the model.
    """

    input_layer = features['board']

    inception_module_outputs, activation_summaries = build_convolutional_modules(
        input_layer,
        params['inception_modules'],
        mode,
        params['kernel_initializer'],
        params['kernel_regularizer'],
        params['trainable_cnn_modules'])

    if not params["conv_init_fn"] is None:  # and tf.train.global_step(tf.get_default_session(), tf.train.get_global_step()) == 0:
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

    # Calculate loss (for both TRAIN and EVAL modes)
    if mode != tf.estimator.ModeKeys.PREDICT:
        # Implementing an altered version of the loss function defined in Deep Pink
        # There are a few other methods I've been trying out in commented out, though none seem to be as good as
        # the one proposed in Deep Pink
        with tf.variable_scope("loss"):
            loss = tf.losses.mean_squared_error(labels / 100, tf.squeeze(logits, 1) / 100)

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

    # Generate predictions
    predictions = {"scores": logits}

    # A dictionary for scoring used when exporting model for serving.
    the_export_outputs = {"serving_default": tf.estimator.export.RegressionOutput(value=logits)}

    # Create the validation metrics
    validation_metrics = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        dist_from_desired = logits - labels

        to_create_metric_dict = {
            "metrics/mean_dist_from_desired": dist_from_desired,
            "metrics/mean_abs_dist_from_desired": tf.abs(dist_from_desired),
            "loss/loss": (loss, loss_summary),
        }

        validation_metrics = metric_dict_creator(to_create_metric_dict)

    # Create the trainable variable summaries and merge them together to give to a hook
    trainable_var_summaries = layers.summarize_tensors(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))  # Not sure if needs to be stored as a variable, should check
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


def deep_pink_model_fn(features, labels, mode, params):
    """
    Generates an EstimatorSpec for the model.
    """

    if mode == tf.estimator.ModeKeys.PREDICT:
        input_layer = features["feature"]
    else:
        #Reshape features from original shape of [-1, 3, 8, 8, 16]
        input_layer = tf.reshape(features, [-1, 8, 8, params['num_input_filters']])


    inception_module_outputs, activation_summaries = build_convolutional_modules(
        input_layer,
        params['inception_modules'],
        mode,
        params['kernel_initializer'],
        params['kernel_regularizer'],
        params['trainable_cnn_modules'])


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
            adjusted_equality_sum = (original_pos + desired_pos)
            adjusted_real_rand_sum = (random_pos - desired_pos)

            real_greater_rand_scalar_loss = tf.reduce_mean(-tf.log(tf.sigmoid(adjusted_real_rand_sum)))

            equality_scalar_loss = tf.reduce_mean(-tf.log(tf.sigmoid(adjusted_equality_sum)))
            negative_equality_scalar_loss = tf.reduce_mean(-tf.log(tf.sigmoid( -adjusted_equality_sum)))

            ratio_old_new_sum_loss_to_negative_sum = tf.divide(equality_scalar_loss, negative_equality_scalar_loss)

            real_rand_loss_summary = tf.summary.scalar("real_greater_rand_loss", real_greater_rand_scalar_loss)


            equality_sum_loss_summary = tf.summary.scalar("mean_original_plus_desired_loss", equality_scalar_loss)
            negative_equality_sum_loss_summary = tf.summary.scalar("mean_negative_original_plus_desired", negative_equality_scalar_loss)


            # loss = real_greater_rand_scalar_loss
            loss = real_greater_rand_scalar_loss + equality_scalar_loss + negative_equality_scalar_loss

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


    # Generate predictions
    predictions = {"scores" : logits}

    # A dictionary for scoring used when exporting model for serving.
    the_export_outputs = {"serving_default" : tf.estimator.export.RegressionOutput(value=logits)}


    # Create the validation metrics
    validation_metrics = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        old_plus_desired = original_pos + desired_pos
        rand_real_diff = random_pos - desired_pos

        abs_rand_real_diff = tf.abs(rand_real_diff)

        abs_old_plus_desired = tf.abs(old_plus_desired)
        mean_abs_old_plus_desired = tf.reduce_mean(abs_old_plus_desired)

        abs_randreal_realold_ratio = tf.reduce_mean(rand_real_diff) / mean_abs_old_plus_desired

        rand_vs_real_accuracy = tf.cast(tf.less(desired_pos, random_pos), tf.float32)


        to_create_metric_dict = {
            "metrics/rand_vs_real_accuracy" : rand_vs_real_accuracy,
            "metrics/mean_dist_rand_real" : rand_real_diff,
            "metrics/mean_abs_rand_real_diff" : abs_rand_real_diff,
            "metrics/mean_dist_old_real" : old_plus_desired,
            "metrics/mean_abs_dist_old_real" : mean_abs_old_plus_desired,
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
            "loss/ratio_old_new_sum_loss_to_negative_sum": ratio_old_new_sum_loss_to_negative_sum,

            "loss/loss" : (loss, loss_summary),
        }


        validation_metrics = metric_dict_creator(to_create_metric_dict)

    # Create the trainable variable summaries and merge them together to give to a hook
    trainable_var_summaries = layers.summarize_tensors(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))  # Not sure if needs to be stored as a variable, should check
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


def encoder_builder_fn(features, labels, mode, params):
    """
    Generates an EstimatorSpec for the model.

    NOTES:
    1) I'm not sure yet, but lately I've been thinking the encoder isn't needed/wanted (if the encoder starts being
    used again, this will be refactored immediately)
    """

    if mode == tf.estimator.ModeKeys.PREDICT:
        input_layer = features["data"]
    else:
        input_layer = tf.reshape(features, [-1, 8, 8, params['num_input_filters']])


    logits, activation_summaries = build_convolutional_modules(
        input_layer,
        params['inception_modules'],
        mode,
        params['kernel_initializer'],
        params['kernel_regularizer'],
        params['trainable_cnn_modules'])


    if not params["conv_init_fn"] is None:
        params["conv_init_fn"]()


    loss = None
    legal_move_loss = None
    pieces_loss = None
    train_op = None
    legal_move_summary = None
    pieces_loss_summary = None
    loss_summary = None

    empty_squares = tf.expand_dims(1 - tf.reduce_sum(input_layer, axis=3), axis=3)
    one_hot_piece_labels = tf.concat([input_layer, empty_squares], axis=3)


    piece_logit_slices = logits[..., :16]
    move_logit_slices = logits[..., 16:]


    # Calculate loss (for both TRAIN and EVAL modes)
    if mode != tf.estimator.ModeKeys.PREDICT:
        with tf.variable_scope("loss"):

            index_to_move_dict = {value: key for key, value in bf.generate_move_to_enumeration_dict().items()}
            possible_move_indices = tf.constant([[index_to_move_dict[j][0],index_to_move_dict[j][1]] for j in range(len(index_to_move_dict))], dtype=tf.int32)

            legal_move_ints = tf.transpose(tf.to_int32(labels))

            move_logits_to_from_format = tf.transpose(tf.reshape(move_logit_slices, (-1,64,64)),perm=[1,2,0])

            possible_move_logits = tf.gather_nd(move_logits_to_from_format, possible_move_indices)

            pieces_loss = tf.losses.softmax_cross_entropy(tf.reshape(one_hot_piece_labels, (-1, 16)), tf.reshape(piece_logit_slices, (-1, 16)))
            legal_move_loss = tf.losses.sigmoid_cross_entropy(legal_move_ints, possible_move_logits)

            loss = pieces_loss + legal_move_loss

            pieces_loss_summary = tf.summary.scalar("pieces_loss", pieces_loss)
            legal_move_summary = tf.summary.scalar("legal_move_loss", legal_move_loss)
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


    # Generate predictions
    predictions = {"scores" : logits}

    # A dictionary for scoring used when exporting model for serving.
    the_export_outputs = {"serving_default" : tf.estimator.export.RegressionOutput(value=logits)}

    # Create the validation metrics
    validation_metrics = None
    if mode == tf.estimator.ModeKeys.EVAL:
        piece_predictions = tf.nn.softmax(piece_logit_slices, axis=3)

        calculated_diff = piece_predictions - one_hot_piece_labels

        filter_diff_sums = tf.reduce_sum(calculated_diff, axis=[1, 2])

        mean_abs_diffs = tf.reduce_mean(tf.abs(filter_diff_sums), axis=0)

        to_create_metric_dict = {
            "loss/pieces_loss": (pieces_loss, pieces_loss_summary),
            "loss/legal_move_loss": (legal_move_loss, legal_move_summary),
            "loss/loss": (loss, loss_summary),
            "metrics/mean_abs_ep_diff": mean_abs_diffs[0],
            "metrics/mean_abs_unoccupied_diff": mean_abs_diffs[15],
            "metrics/mean_abs_king_diff": (mean_abs_diffs[1] + mean_abs_diffs[8]) / 2,
            "metrics/mean_abs_queen_diff": (mean_abs_diffs[2] + mean_abs_diffs[9]) / 2,
            "metrics/mean_abs_not_castling_rook_diff": (mean_abs_diffs[3] + mean_abs_diffs[10]) / 2,
            "metrics/mean_abs_bishop_diff": (mean_abs_diffs[4] + mean_abs_diffs[11]) / 2,
            "metrics/mean_abs_knight_diff": (mean_abs_diffs[5] + mean_abs_diffs[12]) / 2,
            "metrics/mean_abs_pawn_diff": (mean_abs_diffs[6] + mean_abs_diffs[13]) / 2,
            "metrics/mean_abs_can_castle_rook_diff": (mean_abs_diffs[7] + mean_abs_diffs[14]) / 2,
        }


        validation_metrics = metric_dict_creator(to_create_metric_dict)


    # Create the trainable variable summaries and merge them together to give to a hook
    trainable_var_summaries = layers.summarize_tensors(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))  # Not sure if needs to be stored as a variable, should check
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
    def numpy_style_repeat_1d(input, multiples):
        tiled_input = tf.multiply(tf.ones([100, 1]), input)
        return tf.boolean_mask(tiled_input, tf.sequence_mask(multiples))


    inception_module_outputs, activation_summaries = build_convolutional_modules(
        features["board"],
        params['inception_modules'],
        mode,
        params['kernel_initializer'],
        params['kernel_regularizer'],
        params['trainable_cnn_modules'])


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


    legal_move_logits = tf.gather_nd(logits, features["legal_move_indices"])

    # Calculate loss (for both TRAIN and EVAL modes)
    if mode != tf.estimator.ModeKeys.PREDICT:
        with tf.variable_scope("loss"):
            loss = tf.losses.mean_squared_error(legal_move_logits, features["desired_scores"])
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
    predictions = {"the_move_values": logits}

    # A dictionary for scoring used when exporting model for serving.
    the_export_outputs = {"serving_default": tf.estimator.export.ClassificationOutput(scores=legal_move_logits)}


    # Create the validation metrics
    validation_metrics = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        calculated_best_move_scores = tf.gather(legal_move_logits, features['desired_move_indices'])

        repeated_best_scores = numpy_style_repeat_1d(calculated_best_move_scores, features['num_moves'])

        ratio_moves_below_best = tf.reduce_mean(tf.cast(tf.greater_equal(repeated_best_scores, legal_move_logits), dtype=np.float32))

        diff_from_desired = legal_move_logits - features["desired_scores"]


        mean_diff_from_desired = tf.reduce_mean(diff_from_desired)
        mean_calculated_value = tf.reduce_mean(legal_move_logits)

        to_create_metric_dict = {
            "loss/loss" : (loss, loss_scalar_summary),
            "metrics/ratio_moves_below_best" : ratio_moves_below_best,
            "metrics/mean_evaluation_value" : mean_calculated_value,
            "metrics/mean_abs_evaluation_value": tf.abs(legal_move_logits),
            "metrics/mean_expected_value" : features["desired_scores"],
            "metrics/mean_abs_expected_value" : abs(features["desired_scores"]),
            "metrics/distance_from_desired": mean_diff_from_desired,
            "metrics/abs_distance_from_desired" : tf.abs(diff_from_desired),
            "metrics/relative_distance_from_desired": tf.abs(mean_diff_from_desired / mean_calculated_value),
        }

        validation_metrics = metric_dict_creator(to_create_metric_dict)


    # Create the trainable variable summaries and merge them together to give to a hook
    trainable_var_summaries = layers.summarize_tensors(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))  # Not sure if needs to be stored as a variable, should check
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


def one_hot_create_tf_records_input_data_fn(filename_pattern, batch_size=None, include_unoccupied=True,
                                            shuffle_buffer_size=None, repeat=True, num_things_in_parallel=12,
                                            return_before_prefetch=False):
    def tf_records_input_data_fn():
        filenames = tf.data.Dataset.list_files(filename_pattern)
        dataset = filenames.apply(
            tf.contrib.data.parallel_interleave(
                lambda filename : tf.data.TFRecordDataset(filename),
                cycle_length=5,
                sloppy=True))

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
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

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
                                   shuffle_buffer_size=None, repeat=True, num_things_in_parallel=12, num_things_to_prefetch=None):

    if num_things_to_prefetch is None:
        num_things_to_prefetch = num_things_in_parallel



    dataset_1 = one_hot_create_tf_records_input_data_fn(filename_pattern, None, include_unoccupied,
                                                        shuffle_buffer_size, repeat, num_things_in_parallel,
                                                        return_before_prefetch=True)()

    dataset_2 = one_hot_create_tf_records_input_data_fn(filename_pattern, None, include_unoccupied,
                                                        shuffle_buffer_size, repeat, num_things_in_parallel,
                                                        return_before_prefetch=True)()


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


def encoder_tf_records_input_data_fn(filename_pattern, batch_size, shuffle_buffer_size=100000, include_unoccupied=True,
                                     repeat=True, shuffle=True, num_things_in_parallel=12):
    """
    NOTES:
    1) I'm not sure yet, but lately I've been thinking the encoder isn't needed/wanted (if the encoder starts being
    used again, this will be refactored immediately)
    """
    def tf_records_input_data_fn():

        filenames = tf.data.Dataset.list_files(filename_pattern)
        dataset = filenames.apply(
            tf.contrib.data.parallel_interleave(
                lambda filename: tf.data.TFRecordDataset(filename),
                cycle_length=5,
                sloppy=True,
            ))

        def parser(record):
            context_features = {"board": tf.FixedLenFeature([64], tf.int64)}
            sequence_features = {"moves": tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)}

            parsed_record = tf.parse_single_sequence_example(record, context_features=context_features,sequence_features=sequence_features)

            if include_unoccupied:
                reshaped_board = tf.one_hot(tf.reshape(parsed_record[0]["board"],[8,8]),16)
            else:
                reshaped_board = tf.one_hot(tf.reshape(parsed_record[0]["board"], [8, 8])-1,15)

            sparse_tensor = tf.sparse_tensor_to_dense(tf.SparseTensor(tf.expand_dims(
                parsed_record[1]["moves"], 1),tf.fill(tf.shape(parsed_record[1]["moves"]),True), [1792]),
                default_value=False,validate_indices=False)

            return reshaped_board, sparse_tensor

        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

        dataset = dataset.map(parser, num_parallel_calls=num_things_in_parallel)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(num_things_in_parallel)

        if repeat:
            dataset = dataset.repeat()

        iterator = dataset.make_one_shot_iterator()

        features = iterator.get_next()

        return features[0], features[1]

    return tf_records_input_data_fn


def move_index_getter(moves, moves_per_board, max_batch_size=20000, max_moves_per_board=100):
    board_index_repeated_array = tf.transpose(
        tf.reshape(
            tf.tile(tf.range(max_moves_per_board), [max_batch_size]),
            [max_batch_size, max_moves_per_board]),
        [1, 0])

    move_to_index_tensor = tf.constant(bf.MOVE_TO_INDEX_ARRAY, shape=[64, 64])

    board_indices_for_moves = tf.boolean_mask(board_index_repeated_array,
                                              tf.sequence_mask(tf.cast(moves_per_board, tf.int32)))

    move_nums = tf.gather_nd(move_to_index_tensor, tf.cast(moves, tf.int32))

    return tf.stack([board_indices_for_moves, move_nums], axis=-1)



def move_gen_create_tf_records_input_data_fn(filename_pattern, batch_size, shuffle_buffer_size=100000,
                                             include_unoccupied=True, repeat=True, shuffle=True,
                                             num_things_in_parallel=12, max_moves_per_board=100):
    def input_fn():
        filenames = tf.data.Dataset.list_files(filename_pattern)
        dataset = filenames.apply(
            tf.contrib.data.parallel_interleave(
                lambda filename: tf.data.TFRecordDataset(filename),
                cycle_length=5,
                sloppy=True,
            ))


        def parser(records):
            features = {
                "board": tf.FixedLenFeature([64], tf.int64),
                "from_squares" : tf.VarLenFeature(tf.int64),
                "to_squares": tf.VarLenFeature(tf.int64),
                "move_scores": tf.VarLenFeature(tf.float32),
                "num_moves" : tf.FixedLenFeature([], tf.int64)
            }

            parsed_record = tf.parse_example(records, features)

            to_from_squares = tf.stack([parsed_record["from_squares"].values,
                                        parsed_record["to_squares"].values], axis=1)

            move_lookup_indices = move_index_getter(to_from_squares,
                                                    parsed_record['num_moves'],
                                                    batch_size,
                                                    max_moves_per_board)

            reshaped_board = tf.reshape(parsed_record["board"], [-1, 8, 8])

            if include_unoccupied:
                one_hot_board = tf.one_hot(reshaped_board, 16)
            else:
                one_hot_board = tf.one_hot(reshaped_board - 1, 15)

            dense_desired_moves = tf.sparse_tensor_to_dense(parsed_record['move_scores'], default_value=tf.float32.min)
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


def board_eval_serving_input_receiver():
    (piece_bbs, color_occupied_bbs, ep_squares, castling_lookup_indices, kings), formatted_data = get_board_data()

    receiver_tensors = {"piece_bbs": piece_bbs,
                        "color_occupied_bbs": color_occupied_bbs,
                        "ep_squares": ep_squares,
                        "castling_lookup_indices": castling_lookup_indices,
                        "kings": kings}

    dict_for_model_fn = {"board": formatted_data}

    return tf.estimator.export.ServingInputReceiver(dict_for_model_fn, receiver_tensors)


def move_scoring_serving_input_receiver(max_batch_size=50000, max_moves_per_board=100):
    (piece_bbs, color_occupied_bbs, ep_squares, castling_lookup_indices, kings), formatted_data = get_board_data()

    moves_per_board = tf.placeholder(tf.uint8, shape=[None], name="moves_per_board_placeholder")
    moves = tf.placeholder(tf.uint8, shape=[None, 2], name="move_placeholder")

    the_moves = move_index_getter(moves, moves_per_board, max_batch_size, max_moves_per_board)

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

