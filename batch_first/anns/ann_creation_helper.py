import tensorflow as tf

from tensorflow.contrib import layers
from tensorflow.python import training

from functools import reduce

from tensorflow.contrib import tensorrt as trt



def save_model_as_graphdef_for_serving(model_path, output_model_path, output_filename, output_node_name,
                                       model_tags="serve", trt_memory_fraction=.4, total_video_memory=1.1e10,
                                       max_batch_size=25000, as_text=False):


    #This would ideally be 1 instead of .7, but the GPU that this is running on is responsible for things like graphics
    with tf.Session(config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.7 - trt_memory_fraction))) as sess:
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


def build_fully_connected_layers_with_batch_norm(the_input, shape, kernel_initializer, mode, scope_prefix=""):
    """
    Builds fully connected layers with batch normalization onto the computational graph of the desired shape.


    The following are a few examples of the shapes that can be given, and how the layers are built

    shape = [512, 256, 128]
    result: input --> 512 --> 256 --> 128

    shape = [[25, 75], [200], 100, 75]
    result: concat((input --> 25 --> 75), (input --> 200)) --> 100 --> 75

    More complicated shapes can be used by defining modules recursively like the following
    shape = [[[100], [50, 50]], [200, 50], [75], 100, 20]
    """
    if len(shape) == 0:
        return the_input

    module_outputs = []
    for j, inner_modules in enumerate(shape):
        if isinstance(inner_modules, list):
            output = build_fully_connected_layers_with_batch_norm(
                the_input,
                inner_modules,
                kernel_initializer,
                mode,
                scope_prefix="%sfc_module_%d/" % (scope_prefix, j + 1))
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
                layers.summarize_activation(the_input)

            module_outputs = [the_input]
            break

    if len(module_outputs) == 1:
        return module_outputs[0]

    return tf.concat(module_outputs, axis=1)


def build_inception_module_with_batch_norm(the_input, module, kernel_initializer, mode,
                                           num_previously_built_inception_modules=0, force_no_concat=False,
                                           make_trainable=True, weight_regularizer=None):
    """
    Builds a convolutional module based on a given design using batch normalization and the rectifier activation.
    It returns the final layer/layers in the module.

    The following are a few examples of what can be used in the 'module' parameter (explanation follows):

    example_1_module = [[[35,1], (1024, 8)]]

    example_2_module = [[[30, 1]],
                        [[15, 1], [30, 3]],
                        [[15, 1], [30, 3, 2]],
                        [[15, 1], [30, 3, 3]],        #  <-- This particular path does a 1x1 convolution on the module's
                        [[10, 1], [20, 3, 4]],        #      input with 15 filters, followed by a 3x3 'same' padded
                        [[8, 1],  [16, 3, 5]],        #      convolution with dilation factor of 3.  It is concatenated
                        [[8, 1],  [16, 2, (2, 4)]],   #      with the output of the other paths, and then returned
                        [[8, 1],  [16, 2, (4, 2)]]]


    :param module: A list (representing the module's shape), of lists (each representing the shape of a 'path' from the
    input to the output of the module), of either tuples or lists of size 2 or 3 (representing individual layers).
    If a tuple is used, it indicates that the layer should use 'valid' padding, and if a list is used it will use a
    padding of 'same'.  The contents of the innermost list or tuple will be the number of filters to create for a layer,
    followed by the information to pass to conv2d as kernel_size, and then optionally, a third element which is
    to be passed to conv2d as a dilation factor.
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

                cur_conv_output = tf.layers.conv2d(
                    inputs=cur_input,
                    filters=section[0],
                    kernel_size=section[1],
                    padding='valid' if isinstance(section, tuple) else 'same',
                    dilation_rate = 1 if len(section) < 3 else section[-1],
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

    list(layers.summarize_activation(layer) for layer in to_summarize)

    with tf.variable_scope("inception_module_" + str(num_previously_built_inception_modules + 1)):
        if len(path_outputs) == 1:
            return path_outputs[0]
        for j in range(1, len(path_outputs)):
            if force_no_concat or path_outputs[0].get_shape().as_list()[1:3] != path_outputs[j].get_shape().as_list()[1:3]:
                return [temp_input for temp_input in path_outputs]

        return tf.concat([temp_input for temp_input in path_outputs], 3)


def build_convolutional_modules(input, modules, mode, kernel_initializer, kernel_regularizer, make_trainable):
    """
    Creates a desired set of convolutional modules.  Primarily the modules are created through the
    build_inception_module_with_batch_norm function, but if it's functionality proves insufficient, a function can be
    given in place of a module shape, which will accept the previous modules output as input, and who's output will
    be given to the next module for input (or returned). (A detailed example is given below, and more information about the
    shape of a module can be found in the comment for the build_inception_module_with_batch_norm function)


    Below is an example of what may be given for the 'modules' parameter, along with a detailed explanation of what
    each component does.


    #Assume the input is such that the following holds true
    tf.shape(input) == [-1, 8, 8, 15]


    modules = [
        [[[20, 3]],           # The following 7 layers with 3x3 kernels and increasing dilation factors are such that
         [[20, 3, 2]],        # their combined filters centered at any given square will consider only the squares in
         [[20, 3, 3]],        # which a queen could possibly attack from, and the central square itself (and padding)
         [[20, 3, 4]],
         [[10, 3, 5]],
         [[10, 3, 6]],
         [[10, 3, 7]],
         [[8, 2, (2, 4)]],     # For any given square, this and the following layer collectively consider all possible
         [[8, 2, (4, 2)]]],    # squares in which a knight could attack from (and padding when needed),

        # After the module's 9 input layers are created, their outputs are concatenated together, and the module's
        # output will have the shape : [-1, 8, 8, 126]


        # Now a module is created which applies a 1x1 convolution, followed by a an 8x8 convolution with valid padding
        [[[32,1], (1024, 8)]],


        # To prepare the outputs from the convolutional modules for the fully connected layers, they are reshaped from
        # rank 4 and shape  [-1, 1, 1, 1024], to rank 2 with shape [-1, 1024]
        lambda x: tf.reshape(x, [-1, 1024]),
    ]

    #The shape [-1, 1024] tensor would then be returned by this function.



    :param modules: A list which sequentially represents the graph operations to be created.  It may contain
     any combination of either functions which accept the previous module's output and return the desired input for
     the next module, or shapes of modules which can be given to the build_inception_module_with_batch_norm method
    """
    cur_inception_module = input

    for module_num, module_shape in enumerate(modules):
        if callable(module_shape):
            cur_inception_module = module_shape(cur_inception_module)
        else:
            cur_inception_module = build_inception_module_with_batch_norm(
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
        return tf.concat(inception_module_paths_flattened, 1)

    return cur_inception_module


def metric_dict_creator(the_dict):
    metric_dict = {}
    for key, value in the_dict.items():
        if isinstance(value, tuple): # Given a tuple (tensor, summary)
            metric_dict[key] = (tf.reduce_mean(value[0]), value[1])
        else: #Given a tensor
            mean_value = tf.reduce_mean(value)
            metric_dict[key] = (mean_value, tf.summary.scalar(key, mean_value))

    return metric_dict


def numpy_style_repeat_1d(input, multiples):
    casted_multiples = tf.cast(multiples, dtype=tf.int32)
    to_return = tf.boolean_mask(
        tf.multiply(tf.ones([tf.reduce_max(casted_multiples), 1], dtype=input.dtype), input),
        tf.sequence_mask(casted_multiples))
    return to_return


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

    def __init__(self, step_increment, estimator, input_fn_creator, temp_num_steps_in_epoch=None,
                 recall_input_fn_creator_after_evaluate=False):
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

