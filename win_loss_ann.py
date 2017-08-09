'''
Created on Jul 2, 2017

@author: Samuel Ragusa
'''
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
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


def cnn_model_fn(features, labels, mode, params):
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
        path_outputs = [None for _ in range(len(module))]  #See if I can do [None]*len(module) instead
        
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
            #Currently not doing this activaton summary because as of now it is not very helpful.
            #activation_summaries.append(layers.summarize_activation(final_layer))
            
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
        1) Add in the error messages where written in comment
        """
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
        
    
    #for TRAIN_OP_SUMMARIES loss is an option but as of now it is being
    #logged independently so that it can be easily added to another hook.
    TRAIN_OP_SUMMARIES = ["learning_rate", "gradients", "gradient_norm"]
    THE_OPTIMIZER = params['optimizer']
    NUM_OUTPUT_NEURONS = params['num_outputs']
    INCEPTION_1_SHAPE = params['inception_module1']
    INCEPTION_2_SHAPE = params['inception_module2']
    DENSE_LAYERS_SHAPE = params['dense_shape']
    DENSE_LAYERS_DROPOUT_RATES = params['dense_dropout']
    LEARNING_RATE = params['learning_rate']
    SAVE_SUMMARY_INTERVAL = params['log_interval']
    SAVE_MODEL_DIR = params['model_dir']

    #Reshaped the input data to be like a chess board (8x8) 
    input_layer = tf.reshape(features, [-1,8,8,1])

    #Build the first inception module
    inception_module1, activation_summaries = build_inception_module(input_layer, INCEPTION_1_SHAPE)
    
    #Build the second inception module
    inception_module2, activation_summaries = build_inception_module(inception_module1, INCEPTION_2_SHAPE, activation_summaries,1)

    #1300 is the sum of the number of features in the last
    #section of each "path" in the inception layer.
    #The 8s are the dimensions of each feature map
    FIGURE_THIS_OUT_AS_FUNCTION_OF_CONSTANTS = 1300*8*8  
    
    #Reshape the output from the convolutional layers for the densely connected layers
    flat_conv_output = tf.reshape(inception_module2, [-1, FIGURE_THIS_OUT_AS_FUNCTION_OF_CONSTANTS])
    
    #Build the fully connected layers 
    dense_layers_outputs = build_fully_connected_layers(
        flat_conv_output,
        DENSE_LAYERS_SHAPE,
        DENSE_LAYERS_DROPOUT_RATES)
    
    
    #Create the final layer of the ANN
    logits = tf.layers.dense(inputs=dense_layers_outputs, units=NUM_OUTPUT_NEURONS, name="FC_" + str(len(DENSE_LAYERS_SHAPE)+1))
    
    
    #Figure out if these are needed.  Don't think so, but don't have time to think about it right now
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
    summary_hook = tf.train.SummarySaverHook(save_steps = SAVE_SUMMARY_INTERVAL, output_dir =SAVE_MODEL_DIR, summary_op=merged_summaries)
    
    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, 
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=[summary_hook])
    


def main(unused_param):
    """
    Set up the data pipelines, create the computational graph, train the model, and evaluate the results.
    """
    
    def accuracy_metric(predictions, labels,weights=None):
        """
        An accuracy metric built to be able to given to tf.contrib.learn.MetricSpec
        as a metric.
        
        This is only used because the TensorFlow built in metrics haven't been working.
        """
        return tf.reduce_mean(tf.cast(tf.equal(predictions,tf.argmax(labels, 1)),tf.float32))
    
    
    def input_data_fn(data_getter_ops):
        """
        The function which is called to get data for the fit functSion.
        
        NOTES:
        1) Need to build a better method for keeping track of threads.  This is
        very important for running over multiple epochs.
        """
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
        batch, labels = sess.run(data_getter_ops)
#         coord.request_stop()
#         coord.join(threads)
        
        #See if these tf.constant() calls are needed or if I should have them be computed elsewhere
        return tf.constant(batch, dtype=tf.float32), tf.constant(labels, dtype=tf.float32)
    
    #Hopefully set these constants up with something like flags to better be able
    #to run modified versions of the model in the future (when tuning the model)
    SAVE_MODEL_DIR = "/tmp/win_loss_ann"
    TRAINING_DATA_FILENAME = "train_data.csv"
    VALIDATION_FILENAME = "validation_data.csv"
    TESTING_FILENAME = "test_data.csv"
    NUM_OUTPUTS = 2
    DENSE_SHAPE = [500]#[2048,4096,512]
    DENSE_DROPOUT = .4   #NEED TO PICK THIS PROPERLY
    OPTIMIZER = "SGD"    #NEED TO PICK THIS PROPERLY
    TRAINING_MIN_AFTER_DEQUEUE = 1500      
    TESTING_MIN_AFTER_DEQUEUE = 2000
    TRAINING_BATCH_SIZE = 500  #NEED TO PICK THIS PROPERLY
    TESTING_BATCH_SIZE = 1000
    NUM_TESTING_STEPS = 200       #Should be set to number_of_testing_examples//TESTING_BATCH_SIZE
    NUM_EPOCHS =  2000     #I think where I'm passing this parameter to isn't actually epochs, it's batches
    LOG_ITERATION_INTERVAL = 100
    INCEPTION_MODULE_1_SHAPE = [[[200,2]],[[300,3]],[[500,5]]]
    INCEPTION_MODULE_2_SHAPE = [[[500,1]],[[500,1],[300,3]],[[500,1],[500,5]]]
    LEARNING_RATE = .001   #NEED TO PICK THIS PROPERLY
    
    #Create training data pipeline
    training_data_pipe_ops = data_pipeline(    #MAKE SURE THAT THIS WILL WORK WHEN DONE ON MANY ITERATIONS (because I'm pretty sure it won't)
        [TRAINING_DATA_FILENAME],
        TRAINING_BATCH_SIZE, 
        min_after_dequeue=TRAINING_MIN_AFTER_DEQUEUE)
    
    #Create testing data pipeline
    testing_data_pipe_ops = data_pipeline(
        [TESTING_FILENAME],
        TESTING_BATCH_SIZE,
        min_after_dequeue=TRAINING_MIN_AFTER_DEQUEUE)

    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    

    #Create a Config object such that during training the saves and logs will be done at the intervals I want
    the_config = tf.estimator.RunConfig().replace(save_checkpoints_steps=LOG_ITERATION_INTERVAL, save_summary_steps=LOG_ITERATION_INTERVAL)
    
    
    #Create the Estimator
    classifier = learn.Estimator(
        model_fn=cnn_model_fn,
        model_dir=SAVE_MODEL_DIR,
        config = the_config,
        params = {
            'dense_shape': DENSE_SHAPE,
            'dense_dropout' : DENSE_DROPOUT,
            'optimizer' : OPTIMIZER,
            'num_outputs' : NUM_OUTPUTS,
            'log_interval': LOG_ITERATION_INTERVAL,
            'model_dir' : SAVE_MODEL_DIR,
            'inception_module1' : INCEPTION_MODULE_1_SHAPE,
            'inception_module2' : INCEPTION_MODULE_2_SHAPE,
            "learning_rate": LEARNING_RATE})
    
    
    #Set up testing of validation data at a given interval
#     validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
#         VALIDATION_DATA,
#         VALIDATION_LABELS,
# #         metrics = validation_metrics,
#         every_n_steps=LOG_ITERATION_INTERVAL)
    

    #fit the model to the training data given 
    classifier.fit(
        input_fn=lambda: input_data_fn(training_data_pipe_ops),
        steps=NUM_EPOCHS)

    #Configure the accuracy metric for evaluation 
    testing_metrics = {
        "prediction accuracy": learn.MetricSpec(
            metric_fn= accuracy_metric,
            prediction_key="classes")}

    #Evaluate the model and print results
    eval_results = classifier.evaluate(
        input_fn=lambda: input_data_fn(testing_data_pipe_ops),
        metrics=testing_metrics,
        steps=NUM_TESTING_STEPS)  #steps is here so that the data pipeline doesn't make it hang
    
    coord.request_stop()
    coord.join(threads)
    
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()