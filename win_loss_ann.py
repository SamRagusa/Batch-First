'''
Created on Jul 2, 2017

@author: Samuel Ragusa
'''
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers   #different than the tf.layers methods called elsewhere
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib



tf.logging.set_verbosity(tf.logging.INFO)


sess = tf.InteractiveSession()


def data_pipeline(filenames, batch_size, num_epochs=None, min_after_dequeue=10000):
    """
    Creates a pipeline for the data contained within the given files.  
    It does this using TensorFlow queues.
    
    @return: A tuple in which the first element is a graph operation
    which gets a random batch of the data, and the second element is
    a graph operation to get a batch of the labels corresponding
    to the data gotten by the first element of the tuple.
    
    NOTES:
    1) This should be done on the CPU while the GPU handles the training
    and evaluation.
    """
    with tf.name_scope("data_pipeline"):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
        
        reader = tf.TextLineReader()
        _, value = reader.read(filename_queue)
        
        row = tf.decode_csv(value, record_defaults=[[0.0] for _ in range(66)])
        
        #change the row[:len(row)-2] type stuff to be more pythonic
        example_op, label_op  = tf.stack(row[:len(row)-2]), tf.stack(row[len(row)-2:])
        
        capacity = min_after_dequeue + 3 * batch_size 
        
        example_batch, label_batch = tf.train.shuffle_batch(
            [example_op, label_op],
            batch_size=batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        
        return example_batch, label_batch


def cnn_model_fn(features, labels, mode):
    """
    Model function for the CNN.
    """
    
    def build_inception_module(the_input, module, activation_summaries=[], num_previously_built_inception_modules=0):
        """
        Builds an inception module based on the design given to the function.  It returns the final layer in the module,
        and the activation summaries generated for the layers within the inception module.
        
        The layers will be named "module_N_path_M/layer_P", where N is the inception module number, M is what path number it is on,
        and P is what number layer it is in that path.      
        
        Module of the format:
        [[[filters1_1,kernal_size1_1],... , [filters1_M,kernal_size1_M]],... ,
            [filtersN_1,kernal_sizeN_1],... , [filtersN_P,kernal_sizeN_P]]
            
        (MAYBE IMPLEMENT) A set of [filtersJ_I, kernal_sizeJ_I] can be replaced by None for passing of input to concatination
        """
        path_outputs = [None for _ in range(len(module))]  ##SEE if I can do [None]*len(module) instead
        
        cur_input = None
        for j, path in enumerate(module):
            with tf.name_scope("inception_module_" + str(num_previously_built_inception_modules + 1) + "_path_" + str(j+1)):
                for i, section in enumerate(path):
                    if i==0:
                        if j != 0:
                            path_outputs[j-1] = cur_input
                            
                        cur_input = the_input
                    
                    cur_input = tf.layers.conv2d(
                        inputs=cur_input,
                        filters=section[0],
                        kernel_size=[section[1], section[1]],
                        padding="same",
                        activation=tf.nn.relu,
                        name="inception_module_" + str(num_previously_built_inception_modules+1) + "_path_" + str(j+1) + "/layer_" + str(i+1))
                    
                    activation_summaries.append(layers.summarize_activation(cur_input))
                    
        path_outputs[-1] = cur_input
        
        with tf.name_scope("inception_module_" + str(num_previously_built_inception_modules + 1) + "_concat"):
            final_layer = tf.nn.relu(tf.concat([temp_input for temp_input in path_outputs],3), name="inception_module_" + str(num_previously_built_inception_modules + 1) + "_concat")
            activation_summaries.append(layers.summarize_activation(final_layer))
            
        return final_layer, activation_summaries
    
    
    def build_fully_connected_layers(the_input, shape, dropout_rates=None, num_previous_fully_connected_layers=0):
        """
        a function to build the fully connected layers onto the computational graph from
        given specifications.
        
        Dropout_rates if kept as the default None will be 0 for every layer, if set to a scaler
        between 0 (inclusive) and 1 (exclusive) will apply the given dropout rate to every layer 
        being built, lastly dropout_rates can be an array the same size as the shape parameter,
        where the dropout rate at index j will be applied to the layer with shape given at index
        j of the shape parameter.  
        
        shape of the format:
        [num_neurons_layer_1,num_neurons_layer_2,...,num_neurons_layer_n]
        
        NOTES:
        1) Make sure altering the_input doesn't have a pointer related problem
        2) Add in the error messages where written in comment
        """
        
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$CONSIDER PUTTING THIS WHOLE SECTION UNDER ONE name_scope$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        with tf.name_scope("FC_" + str(num_previous_fully_connected_layers + 1)):
            if dropout_rates == None:
                dropout_rates = [0]*len(shape)
            elif not isinstance(dropout_rates, list):
                if dropout_rates >= 0 and dropout_rates < 1:
                    dropout_rates = [dropout_rates]*len(shape)
                else:
                    #PRINT THE ERROR BETTER
                    print("THIS ERROR NEEDS TO BE HANDLED BETTER!   1")
            else:
                if len(dropout_rates) != len(shape):
                    #PRINT THE ERROR BETTER
                    print("THIS ERROR NEEDS TO BE HANDLED BETTER!   2")
                    
            
            for index, size, dropout in zip(range(len(shape)),shape, dropout_rates):
                #Figure out if instead I should use tf.layers.fully_connected
                temp_layer_dense = tf.layers.dense(inputs=the_input, units=size, activation=tf.nn.relu, name="FC_" + str(index + num_previous_fully_connected_layers + 1))
                if dropout != 0:
                    the_input =  tf.layers.dropout(inputs=temp_layer_dense, rate=dropout, training=mode == learn.ModeKeys.TRAIN)
                else:
                    the_input = temp_layer_dense
            
            return the_input
        
    
    #set these constants up with something like flags to better be able
    #to run modified versions of the model in the future (when tuning the model)
    #
    #for TRAIN_OP_SUMMARIES loss is an option but as of now it is being
    #logged independently so that it can be easily added to another hook.
    TRAIN_OP_SUMMARIES = ["learning_rate", "gradients", "gradient_norm"]
    THE_OPTIMIZER = "SGD"
    NUM_OUTPUT_NEURONS = 2
    INCEPTION_1_SHAPE = [[[200,2]],[[300,3]],[[500,5]]]
    INCEPTION_2_SHAPE = [[[500,1]],[[500,1],[300,3]],[[500,1],[500,5]]]
    DENSE_LAYERS_SHAPE = [2048,4096,512]
    DENSE_LAYERS_DROPOUT_RATES = .4
    LEARNING_RATE = .001
    SAVE_SUMMARY_INTERVAL = 5

    
    #activation_summaries = []
    
    input_layer = tf.reshape(features, [-1,8,8,1])

    inception_module1, activation_summaries = build_inception_module(input_layer, INCEPTION_1_SHAPE)
    
    inception_module2, activation_summaries = build_inception_module(inception_module1, INCEPTION_2_SHAPE, activation_summaries,1)

    #1300 is the sum of the number of features in the last
    #section of each "path" in the inception layer.
    #The 8s are the dimensions of each feature map
    FIGURE_THIS_OUT_AS_FUNCTION_OF_CONSTANTS = 1300*8*8  
    
    flat_conv_output = tf.reshape(inception_module2, [-1, FIGURE_THIS_OUT_AS_FUNCTION_OF_CONSTANTS])
    dense_layers_outputs = build_fully_connected_layers(
        flat_conv_output,
        DENSE_LAYERS_SHAPE,
        DENSE_LAYERS_DROPOUT_RATES)
    
    
    logits = tf.layers.dense(inputs=dense_layers_outputs, units=NUM_OUTPUT_NEURONS, name="FC_" + str(len(DENSE_LAYERS_SHAPE)+1))
    
    loss = None
    train_op = None
    
    # Calculate loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=labels, logits=logits)
        tf.summary.scalar('loss_summary', loss)
        
    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=LEARNING_RATE,
            optimizer=THE_OPTIMIZER,
            summaries = TRAIN_OP_SUMMARIES)
        
    # Generate Predictions
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(
        logits, name="softmax_tensor")
    }
    
    #Create the trainable variable summaries and merge them together to give to a hook
    trainable_var_summaries = layers.summarize_tensors(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))  #Not sure if needs to be set as a variable, should check
    merged_summaries = tf.summary.merge_all()
    summary_hook = tf.train.SummarySaverHook(save_steps = SAVE_SUMMARY_INTERVAL, output_dir ="/tmp/win_loss_ann", summary_op=merged_summaries)
    
    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, 
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=[summary_hook])
    

def main(unused_param):
    """
    Sets up the data pipeline, creates the computational graph, and trains the model.
    """
    
    #set these constants up with something like flags to better be able
    #to run modified versions of the model in the future (when tuning the model)
    TRAINING_DATA_FILENAME = "tensor_flow_input_database.csv"
    SAVE_MODEL_DIR = "/tmp/win_loss_ann123"
    VALIDATION_DATA = None              #Need to load these up, likely should keep them in RAM because it'd be quick
    VALIDATION_LABELS = None            #Need to load these up, likely should keep them in RAM because it'd be quick
    TEST_DATA = None                    #Need to load these up, likely should keep them in RAM because it'd be quick
    TEST_LABELS = None                  #Need to load these up, likely should keep them in RAM because it'd be quick
    BATCH_SIZE = 500  #NEED TO SET THIS
    MIN_AFTER_DEQUEUE = 1000
    NUM_EPOCHS = 200
    PRINT_ITERATION_INTERVAL = 2
    SNAPSHOT_INTERVAL = 25
    VALIDATION_INTERVAL = 50
    #LOG_ITERATION_INTERVAL = 2
   
    
    # Load training and eval data
    training_data_pipe_ops = data_pipeline(    #MAKE SURE THAT THIS WILL WORK WHEN DONE ON MANY ITERATIONS
        [TRAINING_DATA_FILENAME],
        BATCH_SIZE, 
        min_after_dequeue=MIN_AFTER_DEQUEUE)
    
    def input_data_fn(data_getter_ops):
        """
        The function which is called to get data for the fit function.
        
        NOTES:
        1) Must make sure that this is not greatly slowing everything down
        2) Should make this and all other data pipeline graph operations
            run on CPU while the GPUs do the heavy math operations
        3) Figure out if I can start the coordinator and the queue runners
            and continue getting batches until the epoch is complete, then 
            request_stop() and join the thread.  Need to reread the queue_runner
            guides on TensorFlow's website.
        """
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        batch, labels = sess.run(data_getter_ops)
        coord.request_stop()
        coord.join(threads)
        
        return tf.constant(batch, dtype=tf.float32), tf.constant(labels, tf.int32)
        
    
    # Create the Estimator
    classifier = learn.Estimator(
        model_fn=cnn_model_fn,
        model_dir=SAVE_MODEL_DIR)
    
    checkpoint_saver = tf.train.CheckpointSaverHook(
        SAVE_MODEL_DIR,
        save_steps=SNAPSHOT_INTERVAL)
    
    # Set up logging for predictions
    tensors_to_print = {"loss": "softmax_cross_entropy_loss/value:0"}
    print_logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_print,
        every_n_iter=PRINT_ITERATION_INTERVAL)
    
    #Set up testing of validation data at a given interval
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        VALIDATION_DATA,
        VALIDATION_LABELS,
        every_n_steps=VALIDATION_INTERVAL)
    
    classifier.fit(
        input_fn=lambda: input_data_fn(training_data_pipe_ops),
        steps=NUM_EPOCHS,
        monitors=[validation_monitor, print_logging_hook, checkpoint_saver])
    
    
    # Configure the accuracy metric for evaluation
    metrics = {
        "accuracy": learn.MetricSpec(
            metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }
    
    # Evaluate the model and print results
    eval_results = classifier.evaluate(
        x=TEST_DATA,
        y=TEST_LABELS,
        metrics=metrics)
    
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()