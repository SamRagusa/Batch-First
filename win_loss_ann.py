'''
Created on Jul 2, 2017

@author: SamRagusa
'''
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python import training

from functools import reduce


tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode, params):
    """
    Model function for the CNN.
    """
    
    def build_inception_module(the_input, module, activation_summaries=[], num_previously_built_inception_modules=0, padding='same'):
        """
        Builds an inception module based on the design given to the function.  It returns the final layer in the module,
        and the activation summaries generated for the layers within the inception module.
        
        The layers will be named "module_N_path_M/layer_P", where N is the inception module number, M is what path number it is on,
        and P is what number layer it is in that path.

        Module of the format:
        [[[filters1_1,kernal_size1_1],... , [filters1_M,kernal_size1_M]],... ,
            [filtersN_1,kernal_sizeN_1],... , [filtersN_P,kernal_sizeN_P]]
        
        (MAYBE IMPLEMENT) A set of [filtersJ_I, kernal_sizeJ_I] can be replaced by None for passing of input to concatenation
        """
        path_outputs = [None for _ in range(len(module))]  # See if I can do [None]*len(module) instead
        to_summarize = []
        cur_input = None
        for j, path in enumerate(module):
            with tf.variable_scope("inception_module_" + str(num_previously_built_inception_modules + 1) + "_path_" + str(j + 1)):
                for i, section in enumerate(path):
                    if i == 0:
                        if j != 0:
                            path_outputs[j - 1] = cur_input

                        cur_input = the_input
                    
                    cur_input = tf.layers.conv2d(
                        inputs=cur_input,
                        filters=section[0],
                        kernel_size=[section[1], section[1]],
                        padding=padding,
                        activation=tf.nn.relu,
                        kernel_initializer=layers.xavier_initializer(),
                        bias_initializer=tf.constant_initializer(params['init_bias_value'], dtype=tf.float32),
                        name="layer_" + str(i + 1))

                    to_summarize.append(cur_input)
                    
        path_outputs[-1] = cur_input
        
        activation_summaries = activation_summaries + [layers.summarize_activation(layer) for layer in to_summarize]
        
        with tf.variable_scope("inception_module_" + str(num_previously_built_inception_modules + 1)):
            for j in range(1, len(path_outputs)):
                if path_outputs[0].get_shape().as_list()[1:3] != path_outputs[j].get_shape().as_list()[1:3]:
                    return [temp_input for temp_input in path_outputs], activation_summaries

            return tf.concat([temp_input for temp_input in path_outputs], 3), activation_summaries


    def build_fully_connected_layers(the_input, shape, dropout_rates=None, num_previous_fully_connected_layers=0, activation_summaries=[]):
        """
        a function to build the fully connected layers onto the computational graph from
        given specifications.
        
        Dropout_rates if kept as the default None will be 0 for every layer, if set to a scalar
        between 0 (inclusive) and 1 (exclusive) will apply the given dropout rate to every layer 
        being built, lastly dropout_rates can be an array the same size as the shape parameter,
        where the dropout rate at index j will be applied to the layer with shape given at index
        j of the shape parameter.
        
        shape of the format:
        [num_neurons_layer_1,num_neurons_layer_2,...,num_neurons_layer_n]
        
        NOTES:
        1) Add in the error messages where written in comment
        """
        
        if dropout_rates == None:
            dropout_rates = [0] * len(shape)
        elif not isinstance(dropout_rates, list):
            if dropout_rates >= 0 and dropout_rates < 1:
                dropout_rates = [dropout_rates] * len(shape)
            else:
                print("THIS ERROR NEEDS TO BE HANDLED BETTER!   1")
        else:
            if len(dropout_rates) != len(shape):
                print("THIS ERROR NEEDS TO BE HANDLED BETTER!   2")

        for index, size, dropout in zip(range(len(shape)), shape, dropout_rates):
            # Figure out if instead I should use tf.layers.fully_connected
            with tf.variable_scope("FC_" + str(num_previous_fully_connected_layers + index + 1)):
                temp_layer_dense = tf.layers.dense(
                    inputs=the_input,
                    units=size,
                    activation=tf.nn.relu,
                    kernel_initializer=layers.xavier_initializer(),
                    bias_initializer=tf.constant_initializer(params['init_bias_value'], dtype=tf.float32),
                    name="layer")
                if dropout != 0:
                    the_input = tf.layers.dropout(
                        inputs=temp_layer_dense,
                        rate=dropout,
                        training=mode == tf.estimator.ModeKeys.TRAIN,
                        name="dropout")
                else:
                    the_input = temp_layer_dense
            activation_summaries.append(layers.summarize_activation(temp_layer_dense))
        
        return the_input, activation_summaries


    def dont_choose_random_metric(predictions, the_labels, weights=None):
        """
        NOTES:
        1) Forget why I needed the default parameter weights or the_labels parameter,
        I should check if they're still needed
        """
        accuracy = tf.reduce_mean(tf.cast(tf.greater(predictions[0], predictions[1]), tf.float32))
        return accuracy, tf.summary.scalar("rand_vs_real_accuracy", accuracy)


    def real_move_equal_metric(predictions, the_labels, weights=None):
        """
        NOTES:
        1) Forget why I needed the default parameter weights or the_labels parameter,
        I should check if they're still needed
        """
        difference = tf.reduce_mean(tf.abs(tf.add(predictions[0], predictions[1])))
        return difference, tf.summary.scalar("mean_dist_old_real", difference)


    def move_equal_random_metric(predictions, the_labels, weights=None):
        """
        NOTES:
        1) Same as real_move_equal_metric, but different for logging purposes
        2) Forget why I needed the default parameter weights or the_labels parameter,
        I should check if they're still needed
        """
        difference = tf.reduce_mean(tf.abs(tf.subtract(predictions[0], predictions[1])))
        return difference, tf.summary.scalar("mean_dist_real_rand", difference)


    if mode == tf.estimator.ModeKeys.PREDICT:
        features = features['feature']


    #Probably shouldn't do this during inference (and should let tf.parse or something else handle it)
    input_layer = tf.reshape(features, [-1, 8, 8, 6])


    # Build the first inception module
    inception_module1, activation_summaries = build_inception_module(input_layer, params['inception_module1'], padding='valid')


    #Should REALLY consider converting conv layers to FC layers by doing one convolution with number_of_neurons filters
    inception_module2_paths_flattened = [
        tf.reshape(
            path,
            [-1, reduce(lambda a, b: a * b, path.get_shape().as_list()[1:])]
        ) for path in inception_module1]


    inception_module2_outputs = tf.concat(inception_module2_paths_flattened, 1)


    # Build the fully connected layers
    dense_layers_outputs, activation_summaries = build_fully_connected_layers(
        inception_module2_outputs,
        params['dense_shape'],
        params['dense_dropout'],
        activation_summaries=activation_summaries)

    # Create the final layer of the ANN
    logits = tf.layers.dense(inputs=dense_layers_outputs,
                             units=params['num_outputs'],
                             use_bias=False,
                             activation=None,
                             kernel_initializer=layers.xavier_initializer(),
                             # bias_initializer=tf.constant_initializer(params['init_bias_value'], dtype=tf.float32),
                             name="logit_layer")


    loss = None
    train_op = None

    # (p,q,r) = (original_position, choosen_position, random_position)
    original_pos, desired_pos, random_pos = tf.split(tf.reshape(logits, [-1, 3, 1]),
                                                     [1, 1, 1], 1)

    # Calculate loss (for both TRAIN and EVAL modes)
    if mode != tf.estimator.ModeKeys.PREDICT:

        #Implementing the loss function defined in Deep Pink

        scaled_original_plus_desired = tf.scalar_mul(tf.constant(params['equality_scalar'], dtype=tf.float32), original_pos + desired_pos)

        loss_part_1 = -tf.log(tf.sigmoid(desired_pos - random_pos))
        loss_part_2 = -tf.log(tf.sigmoid(scaled_original_plus_desired))
        loss_part_3 = -tf.log(tf.sigmoid(tf.negative(scaled_original_plus_desired)))  #see if I can use - instead of tf.negative

        loss = tf.reduce_mean(loss_part_1 + loss_part_2 + loss_part_3)


    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params['learning_rate'],
            optimizer=params['optimizer'],
            summaries=params['train_summaries'])


    #Generate predictions (should be None, but it doesn't allow for that)
    # predictions = None
    # if mode != tf.estimator.ModeKeys.PREDICT:
    predictions = {
        "real_rand_guessing": tf.greater(desired_pos, random_pos)}

    # A dictionary for scoring used when exporting model for serving.
    the_export_outputs = {
        "serving_default": tf.estimator.export.RegressionOutput(value=logits)}

    # Create the validation metrics
    validation_metric = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        validation_metric = {
            "validation/rand_real_accuracy": dont_choose_random_metric([desired_pos, random_pos], labels),
            "validation/old_new_diff": real_move_equal_metric([original_pos, desired_pos], labels),
            "validation/real_rand_diff": move_equal_random_metric([desired_pos, random_pos], labels)}

    # Create the trainable variable summaries and merge them together to give to a hook
    trainable_var_summaries = layers.summarize_tensors(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))  # Not sure if needs to be stored as a variable, should check
    merged_summaries = tf.summary.merge_all()
    summary_hook = tf.train.SummarySaverHook(save_steps=params['log_interval'], output_dir=params['model_dir'], summary_op=merged_summaries)

    # Return the EstimatorSpec object
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=[summary_hook],
        export_outputs=the_export_outputs,
        eval_metric_ops=validation_metric)


# Figure out how to use this parameter, and write this method to generate a run
# based on the information given in the parameter
def main(unused_param):
    """
    Set up the data pipelines, create the computational graph, train the model, and evaluate the results.
    """

    def process_line_as_2d_input_serving(board_config):
        """
        This function is likely temporary, and will be deleted when process_line_as_2d_input has
        the capabilities to do what it does now, as well as the changes that have been made in this function.
        """
        with tf.name_scope("process_data_2d"):
            with tf.device("/cpu:0"):
                # A tensor referenced when getting indices of characters for the the_values array
                # NOTE: should make sure this is only being created once per creation of data pipeline
                mapping_strings = tf.constant(["1", "K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"])

                the_values = tf.constant(
                    [[0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1],
                     [-1, 0, 0, 0, 0, 0],
                     [0, -1, 0, 0, 0, 0],
                     [0, 0, -1, 0, 0, 0],
                     [0, 0, 0, -1, 0, 0],
                     [0, 0, 0, 0, -1, 0],
                     [0, 0, 0, 0, 0, -1],
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
                                board_config,
                                delimiter="").values)),
                    [-1, 64, 6])

                return data

    def process_line_as_2d_input(the_str):
        with tf.name_scope("process_data_2d"):
            with tf.device("/cpu:0"):
                # A tensor referenced when getting indices of characters for the the_values array
                # NOTE: should make sure this is only being created once per creation of data pipeline
                mapping_strings = tf.constant(["1", "K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"])

                the_values = tf.constant(
                    [[0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1],
                     [-1, 0, 0, 0, 0, 0],
                     [0, -1, 0, 0, 0, 0],
                     [0, 0, -1, 0, 0, 0],
                     [0, 0, 0, -1, 0, 0],
                     [0, 0, 0, 0, -1, 0],
                     [0, 0, 0, 0, 0, -1],
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
                    [3, 64, 6])

                return data


    def acquire_data_ops(filename_queue, processing_method, record_defaults=None):
        """
        Get the line/lines from the files in the given filename queue,
        read/decode them, and give them to the given method for processing
        the information.
        """
        with tf.name_scope("acquire_data"):
            with tf.device("/cpu:0"):
                if record_defaults is None:
                    record_defaults = [[""]]
                    reader = tf.TextLineReader()
                    key, value = reader.read(filename_queue)
                    row = tf.decode_csv(value, record_defaults=record_defaults)
                    return processing_method(row[0]), tf.constant(True, dtype=tf.bool)


    def data_pipeline(filenames, batch_size, num_epochs=None, min_after_dequeue=10000, allow_smaller_final_batch=False):
        """
        Creates a pipeline for the data contained within the given files.
        It does this using TensorFlow queues.
        
        @return: A tuple in which the first element is a graph operation
        which gets a random batch of the data, and the second element is
        a graph operation to get a batch of the labels corresponding
        to the data gotten by the first element of the tuple.
        
        Notes:
        1) Should likely take in a parameter of the functions for data formatting and processing
        2) Maybe should be using sparse tensors
        3) Should really confirm if min_after_dequeue parameter is referring to examples or batches
        """
        with tf.name_scope("data_pipeline"):
            with tf.device("/cpu:0"):
                filename_queue = tf.train.string_input_producer(filenames, capacity=5000, num_epochs=num_epochs)

                example_op, label_op = acquire_data_ops(filename_queue, process_line_as_2d_input)

                capacity = min_after_dequeue + 3 * batch_size

                example_batch, label_batch = tf.train.shuffle_batch(
                    [example_op, label_op],
                    batch_size=batch_size,
                    capacity=capacity,
                    min_after_dequeue=min_after_dequeue,
                    num_threads=64,
                    allow_smaller_final_batch=allow_smaller_final_batch)
                
                return example_batch, label_batch


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
                allow_smaller_final_batch=allow_smaller_final_batch)
            return batch, labels


    def serving_input_receiver_fn():
        """
        A function to use for input processing when serving the model.
        """
        feature_spec = {'str': tf.FixedLenFeature([1], tf.string)}
        serialized_tf_example = tf.placeholder(dtype=tf.string, name='input_example_tensor')

        receiver_tensors = {'example': serialized_tf_example}

        features = tf.parse_example(serialized_tf_example, feature_spec)

        features['str'] = tf.reshape(features['str'], [-1])
        
        data = process_line_as_2d_input_serving(features['str'])
        # data = tf.map_fn(process_line_as_2d_input_serving, features['str'], dtype=tf.float32)
        
        return tf.estimator.export.ServingInputReceiver(data, receiver_tensors)


    def line_counter(filename):
        """
        A function to count the number of lines in a file efficiently.
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
        
        NOTES:
        1) self._metrics is likely no longer needed since switching from tf.contrib.learn.estimator
        to tf.estimator
        
        TO DO:
        1) Have this not be shuffling batches
        2) Figure out how to handle steps to do one complete epoch
        3) Have this not call the evaluate function because it has to restore from a
        checkpoint, it will likely be faster if I evaluate it on the current training graph
        4) Implement an epoch counter to be printed along with the validation results
        5) Implement some kind of timer so that I can can see how long each epoch takes (printed with results of evaluation)
        """

        def __init__(self, step_increment, estimator, filenames, batch_size=1000, min_after_dequeue=20000, metrics=None,
                     temp_num_steps_in_epoch=None):
            self._step_increment = step_increment
            self._estimator = estimator
            self._filenames = filenames
            self._batch_size = batch_size
            self._min_after_dequeue = min_after_dequeue
            self._metrics = metrics

            # (hopefully) Not permanent
            self._temp_num_steps_in_epoch = temp_num_steps_in_epoch

        def begin(self):
            self._global_step_tensor = training.training_util.get_global_step()
            
            if self._global_step_tensor is None:
                raise RuntimeError("Global step should be created to use ValidationRunHook.")
            
            self._input_fn = lambda: data_pipeline(
                filenames=self._filenames,
                batch_size=self._batch_size,
                num_epochs=None,
                allow_smaller_final_batch=False)  # Ideally this should be True

        def after_create_session(self, session, coord):
            self._step_started = session.run(self._global_step_tensor)

        def before_run(self, run_context):
            return training.session_run_hook.SessionRunArgs(self._global_step_tensor)

        def after_run(self, run_context, run_values):
            if (run_values.results - self._step_started) % self._step_increment == 0:
                print(self._estimator.evaluate(
                    input_fn=self._input_fn,
                    steps=self._temp_num_steps_in_epoch))

    # Hopefully set these constants up with something like flags to better be able
    # to run modified versions of the model in the future (when tuning the model)
    SAVE_MODEL_DIR = "/home/sam/Desktop/tmp/deep_pink_loss_pre_commit_test"
    TRAINING_FILENAMES = ["deep_pink_tester.txt"]
    VALIDATION_FILENAMES = ["deep_pink_tester_leq_5_fics2015.txt"]
    TESTING_FILENAMES = ["new_test_data.csv"]
    TRAIN_OP_SUMMARIES = ["learning_rate", "loss", "gradient_norm", "gradients"]
    NUM_OUTPUTS = 1
    DENSE_SHAPE = [1024, 1024]  # NEED TO PICK THESE PROPERLY
    DENSE_DROPOUT = 0
    INITIAL_BIAS_VALUE = .001
    OPTIMIZER = "SGD"  # 'Adam'
    LOSS_FN = tf.losses.softmax_cross_entropy
    TRAINING_MIN_AFTER_DEQUEUE = 10000
    VALIDATION_MIN_AFTER_DEQUEUE = 10000
    TESTING_MIN_AFTER_DEQUEUE = 1000
    TRAINING_BATCH_SIZE = 500
    EQUALITY_SCALAR_MULT = 1
    VALIDATION_BATCH_SIZE = 2000
    TESTING_BATCH_SIZE = 1000
    NUM_TRAINING_EPOCHS = 500
    LOG_ITERATION_INTERVAL = 5000
    LEARNING_RATE = .0001  # NEED TO PICK THIS PROPERLY
    INPUT_CONV_FILTERS = 400
    INPUT_CONV_KERNAL = [2, 2]
    INCEPTION_MODULE_1_SHAPE = [[[100, 2]], [[100, 3]], [[125, 4]], [[150, 5]]]  # [[[200,2]],[[250,3]],[[350,5]]]
    INCEPTION_MODULE_2_SHAPE = [[[75, 1]], [[100, 1], [50, 3]],[[100, 1], [60, 5]]]  # [[[250,1]],[[250,1],[150,3]],[[250,1],[200,5]]]
    BATCHES_IN_TRAINING_EPOCH = 2000  # reduce(
    # lambda x, y: x + y,
    # [line_counter(filename) for filename in TRAINING_FILENAMES]) // TRAINING_BATCH_SIZE
    BATCHES_IN_VALIDATION_EPOCH = line_counter(VALIDATION_FILENAMES[0]) // VALIDATION_BATCH_SIZE
    # BATCHES_IN_TESTING_EPOCH = line_counter(TESTING_FILENAMES)//TESTING_BATCH_SIZE

    # print(BATCHES_IN_TRAINING_EPOCH)
    # print(BATCHES_IN_VALIDATION_EPOCH)

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=SAVE_MODEL_DIR,
        config=tf.estimator.RunConfig().replace(
            save_checkpoints_steps=LOG_ITERATION_INTERVAL,
            save_summary_steps=LOG_ITERATION_INTERVAL),
        # session_config=tf.ConfigProto(log_device_placement=True)),
        params={
            'dense_shape': DENSE_SHAPE,
            'dense_dropout': DENSE_DROPOUT,
            'optimizer': OPTIMIZER,
            'num_outputs': NUM_OUTPUTS,
            'log_interval': LOG_ITERATION_INTERVAL,
            'model_dir': SAVE_MODEL_DIR,
            'inception_module1': INCEPTION_MODULE_1_SHAPE,
            'inception_module2': INCEPTION_MODULE_2_SHAPE,
            "learning_rate": LEARNING_RATE,
            "train_summaries": TRAIN_OP_SUMMARIES,
            'loss_fn': LOSS_FN,
            'input_conv_filters': INPUT_CONV_FILTERS,
            'input_conv_kernal': INPUT_CONV_KERNAL,
            'init_bias_value': INITIAL_BIAS_VALUE,
            'equality_scalar': EQUALITY_SCALAR_MULT})

    validation_hook = ValidationRunHook(
        step_increment=BATCHES_IN_TRAINING_EPOCH,
        estimator=classifier,
        filenames=VALIDATION_FILENAMES,
        batch_size=VALIDATION_BATCH_SIZE,
        min_after_dequeue=VALIDATION_MIN_AFTER_DEQUEUE,
        temp_num_steps_in_epoch=BATCHES_IN_VALIDATION_EPOCH)

    classifier.train(
        #Could likely just create pipeline in similar way to how I did in ValidationRunHook,
        #eliminating the need for an input function
        input_fn=lambda: input_data_fn(
            TRAINING_FILENAMES,
            TRAINING_BATCH_SIZE,
            NUM_TRAINING_EPOCHS,
            TRAINING_MIN_AFTER_DEQUEUE,
            True),
        # lambda: data_pipeline([TRAINING_FILENAME], TRAINING_BATCH_SIZE, num_epochs, min_after_dequeue))
        hooks=[validation_hook])

    # Export the model for serving
    classifier.export_savedmodel(
        SAVE_MODEL_DIR,
        serving_input_receiver_fn=serving_input_receiver_fn)


#         #Configure the accuracy metric for evaluation
#     testing_metrics = {
#         "prediction_accuracy/testing": learn.MetricSpec(
#             metric_fn= accuracy_metric,
#             prediction_key="classes")}
#
#     #Evaluate the model and print results
#     eval_results = classifier.evaluate(
#         input_fn=lambda: input_data_fn(TESTING_FILENAMES,TESTING_BATCH_SIZE,1,TESTING_MIN_AFTER_DEQUEUE),
#         metrics=testing_metrics,
# #         log_progress=False,
#         steps=BATCHES_IN_TESTING_EPOCH-1)
#
#     print(eval_results)


if __name__ == "__main__":
    tf.app.run()
