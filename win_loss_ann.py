'''
Created on Jul 2, 2017

@author: SamRagusa
'''
from functools import reduce
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


tf.logging.set_verbosity(tf.logging.INFO)


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
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        name="inception_module_" + str(num_previously_built_inception_modules+1) + "_path_" + str(j+1) + "/layer_" + str(i+1))
                    
                    activation_summaries.append(layers.summarize_activation(cur_input))
                    
        path_outputs[-1] = cur_input
        
        with tf.name_scope("inception_module_" + str(num_previously_built_inception_modules + 1) + "_concat"):
            final_layer = tf.nn.relu(tf.concat([temp_input for temp_input in path_outputs],3), name="inception_module_" + str(num_previously_built_inception_modules + 1) + "_concat")
            #Currently not doing this activaton summary because as of now (and maybe forever) it is not very helpful.
            #activation_summaries.append(layers.summarize_activation(final_layer))
            
        return final_layer, activation_summaries
    
    
    def build_fully_connected_layers(the_input, shape, dropout_rates=None, num_previous_fully_connected_layers=0, activation_summaries=[]):
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
                    print("THIS ERROR NEEDS TO BE HANDLED BETTER!   1")
            else:
                if len(dropout_rates) != len(shape):
                    print("THIS ERROR NEEDS TO BE HANDLED BETTER!   2")
                    
            for index, size, dropout in zip(range(len(shape)),shape, dropout_rates):
                #Figure out if instead I should use tf.layers.fully_connected
                temp_layer_dense = tf.layers.dense(
                    inputs=the_input,
                    units=size,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name="FC_" + str(index + num_previous_fully_connected_layers + 1))
                
                activation_summaries.append(layers.summarize_activation(temp_layer_dense))
                if dropout != 0:
                    the_input =  tf.layers.dropout(
                        inputs=temp_layer_dense,
                        rate=dropout,
                        training=mode == learn.ModeKeys.TRAIN)
                else:
                    the_input = temp_layer_dense
            
            return the_input, activation_summaries
        
    

    #Reshaped the input data as a 5d tensor for input to the 3d convolution
    input_layer = tf.reshape(features, [-1,8,8,2,1])  #MAKE SURE THIS WORKS RIGHT

  
    #Create a layer that convolves in 3 dimensions over the input
    #NOTE: I should add TensorBoard integration to this 
    #NOTE: I should make these constants be given by the params dictionary
    conv3d_input_layer = tf.layers.conv3d(
        inputs=input_layer,
        filters=300,
        kernel_size=[2,2,2],  #[depth, height,width]
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        name="conv3d")  
    
    #Reshape the output of the 3 dimensional convolutional layer to a 4d tensor for input to the 2d inception modules
    reshaped_3d_output = tf.reshape(conv3d_input_layer,[-1,7,7,300])#MAKE SURE THIS WORKS RIGHT

    #Build the first inception module
    inception_module1, activation_summaries = build_inception_module(reshaped_3d_output, params['inception_module1'])
    
    #Build the second inception module
    inception_module2, activation_summaries = build_inception_module(inception_module1, params['inception_module2'], activation_summaries,1)
    
    #Reshape the output from the convolutional layers for the densely connected layers
    flat_conv_output = tf.reshape(inception_module2, [-1, reduce(lambda a,b:a*b, inception_module2.get_shape().as_list()[1:]) ])
    
    #Build the fully connected layers 
    dense_layers_outputs, activation_summaries = build_fully_connected_layers(
        flat_conv_output,
        params['dense_shape'],
        params['dense_dropout'],
        activation_summaries=activation_summaries)
    
    
    #Create the final layer of the ANN
    logits = tf.layers.dense(inputs=dense_layers_outputs,
                             units=params['num_outputs'],
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name="logit_layer")
    
    
    #Figure out if these are needed.  Don't think so, but don't have time to think about it right now
    loss = None
    train_op = None
    
    # Calculate loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        loss = params['loss_fn'](
            #Would rather conversion to one_hcot vectors be done within data pipeline
            onehot_labels=tf.one_hot(labels, depth=2), logits=logits)
        
    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params['learning_rate'],
            optimizer=params['optimizer'],
            summaries = params['train_summaries'])
        
    # Generate Predictions
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(
        logits, name="softmax_tensor")
    }
    
    #Create the trainable variable summaries and merge them together to give to a hook
    trainable_var_summaries = layers.summarize_tensors(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))  #Not sure if needs to be set as a variable, should check
    merged_summaries = tf.summary.merge_all()
    summary_hook = tf.train.SummarySaverHook(save_steps = params['log_interval'], output_dir =params['model_dir'], summary_op=merged_summaries)
    
    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, 
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=[summary_hook])
    
    

#Figure out how to use this parameter, and write this method to generate a run
#based on the information given in the parameter
def main(unused_param):
    """
    Set up the data pipelines, create the computational graph, train the model, and evaluate the results.
    """

    def process_line_as_2d_input(row):
        """
        Processes a line of the input text file.  It gets a 64x2 representation
        of the board, and returns it along with the label as a tuple.
        """
        with tf.name_scope("process_data_2d"):
            with tf.device("/cpu:0"):
                #A tensor referenced when getting indices of characters for the the_values array
                #NOTE: should make sure this is only being created once per creation of data pipeline  
                mapping_strings = tf.constant(["1","K","Q","R","B","N","P","k","q","r","b","n","p"])
                
                #An array where each array of 2 values within it represents the value of the 
                #board piece at the same index in mapping_strings
                #NOTE: should make sure this is only being created once per creation of data pipeline
                the_values =  tf.constant(
                    [[0,0],
                    [1,0],
                    [.8,0],
                    [.6,0],
                    [.45,0],
                    [.3,0],
                    [.1,0],
                    [0,1],
                    [0,.8],
                    [0,.6],
                    [0,.45],
                    [0,.3],
                    [0,.1]])
                
                #Create the table for getting indices (for the_values) from the information about the board 
                the_table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings)
                
                #Initialize the table of possibilities for what is in each square
                #NOTE: should make sure this is only being run once per creation of data pipeline
                tf.tables_initializer()
                
                #Using the reshape operation to declare the size of the tensor so it's known
                #(to the computer not to whoever is reading this)
                data = tf.reshape(
                    #Get the values at the given indices
                    tf.gather(
                        the_values,
                        #Get an array of indices corresponding to the array of characters
                        the_table.lookup(
                            #Split the string into an array of characters
                            tf.string_split(
                                [row[0]],
                                delimiter="").values)),
                    [64,2])
        
                return data, row[1]
    
    
    def aquire_data_ops(filename_queue, processing_method):   
        """
        Get the line/lines from the files in the given filename queue,
        read/decode them, and give them to the given method for processing
        the information.
        """
        with tf.name_scope("aquire_data"):
            with tf.device("/cpu:0"):
                reader = tf.TextLineReader()
                key, value = reader.read(filename_queue)
                record_defaults = [[""], [tf.constant(0, dtype=tf.int32)]]
                row = tf.decode_csv(value, record_defaults=record_defaults)
                return processing_method(row)
        
        
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
        3) Should really confirm if min_after_dequeue parameter is refering to examples or batches
        """
        with tf.name_scope("data_pipeline"):
            filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
            
            example_op, label_op = aquire_data_ops(filename_queue, process_line_as_2d_input)
            
            capacity = min_after_dequeue + 3 * batch_size 
            
            example_batch, label_batch = tf.train.shuffle_batch(
                [example_op, label_op],
                batch_size=batch_size,
                capacity=capacity,
                min_after_dequeue=min_after_dequeue,
                allow_smaller_final_batch=allow_smaller_final_batch)
            
            return example_batch, label_batch
    
    
    def input_data_fn(filename, batch_size, epochs, min_after_dequeue, allow_smaller_final_batch=False):
        """
        A function which is called to create the data pipeline ops made by the function 
        data_pipeline.  It does this in a way such that they belong to the session managed 
        by the Estimator object that will be given this function. 
        """
        with tf.name_scope("input_fn"):
            batch, labels =  data_pipeline(
                filenames=[filename],
                batch_size=batch_size,
                num_epochs=epochs,
                allow_smaller_final_batch=allow_smaller_final_batch)
            return batch, labels
      
    
    def accuracy_metric(predictions, labels,weights=None):
        """
        An accuracy metric built to be able to given to tf.contrib.learn.MetricSpec
        as a metric.
        
        This is only used because the TensorFlow built in metrics haven't been working.
        """
        return tf.reduce_mean(tf.cast(tf.equal(predictions,tf.cast(labels,tf.int64)),tf.float32))
    
    
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
    
    #Hopefully set these constants up with something like flags to better be able
    #to run modified versions of the model in the future (when tuning the model)
    SAVE_MODEL_DIR = "/tmp/win_loss_ann_second_fixes1"
    TRAINING_FILENAME = "train_pipeline_eval.csv"#"train_data.csv"
    VALIDATION_FILENAME = "val_pipeline_eval.csv"#"validation_data.csv"
    TESTING_FILENAME = "test_pipeline_eval.csv"#"test_data.csv"
    TRAIN_OP_SUMMARIES = ["learning_rate", "loss", "gradients", "gradient_norm"]
    NUM_OUTPUTS = 2
    DENSE_SHAPE = [4096,512]                                       #NEED TO PICK THIS PROPERLY
    DENSE_DROPOUT = .4                                             #NEED TO PICK THIS PROPERLY
    OPTIMIZER = "SGD"      
    LOSS_FN = tf.losses.softmax_cross_entropy                      #NEED TO PICK THIS PROPERLY
    TRAINING_MIN_AFTER_DEQUEUE = 10000      
    VALIDATION_MIN_AFTER_DEQUEUE = 10000
    TESTING_MIN_AFTER_DEQUEUE = 200
    TRAINING_BATCH_SIZE = 300                                      #NEED TO PICK THIS PROPERLY
    VALIDATION_BATCH_SIZE = 1000
    TESTING_BATCH_SIZE = 100
    NUM_EPOCHS =  10
    LOG_ITERATION_INTERVAL = 300
    LEARNING_RATE = .01                                            #NEED TO PICK THIS PROPERLY
    INCEPTION_MODULE_1_SHAPE = [[[200,2]],[[250,3]],[[350,5]]]
    INCEPTION_MODULE_2_SHAPE = [[[250,1]],[[250,1],[250,3]],[[250,1],[325,5]]]
    BATCHES_IN_VALIDATION_EPOCH = line_counter(VALIDATION_FILENAME)//VALIDATION_BATCH_SIZE
    BATCHES_IN_TESTING_EPOCH = line_counter(TESTING_FILENAME)//TESTING_BATCH_SIZE
    
#     print(line_counter(TRAINING_FILENAME)//TRAINING_BATCH_SIZE)
    
    #Create a Config object such that during training the saves and logs will be done at the intervals I want
    the_config = tf.estimator.RunConfig().replace(save_checkpoints_steps=LOG_ITERATION_INTERVAL, save_summary_steps=LOG_ITERATION_INTERVAL)#,session_config=tf.ConfigProto(log_device_placement=True))
    
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
            "learning_rate": LEARNING_RATE,
            "train_summaries" : TRAIN_OP_SUMMARIES,
            'loss_fn' : LOSS_FN})
    
    #Create the validation metrics to be passed to the classifiers evaluate function
    validation_metrics = {
        "prediction_accuracy/validation": learn.MetricSpec(
            metric_fn= accuracy_metric,
            prediction_key="classes")}
    
    
    
    for j in range(NUM_EPOCHS):
        classifier.fit(  #MAYBE USE PARTIAL FIT
            input_fn=lambda: input_data_fn(TRAINING_FILENAME,TRAINING_BATCH_SIZE,1,TRAINING_MIN_AFTER_DEQUEUE,True))#,
            #lambda: data_pipeline([TRAINING_FILENAME], TRAINING_BATCH_SIZE, num_epochs, min_after_dequeue))
#             max_steps=200)
        
        print("Epoch", str(j+1), "training completed.")
        
        
        #In the long term this should be done by a SessionRunHook to speed up
        #computations by deleting this for loop completely and changing the
        #number of steps to run by NUM_EPOCHS
        validation_results = classifier.evaluate(
            input_fn=lambda: input_data_fn(VALIDATION_FILENAME,VALIDATION_BATCH_SIZE,1,VALIDATION_MIN_AFTER_DEQUEUE),
            metrics=validation_metrics,
            steps=BATCHES_IN_VALIDATION_EPOCH-1,
            log_progress=False)
         
        print(validation_results)
 
    #Configure the accuracy metric for evaluation 
    testing_metrics = {
        "prediction_accuracy/testing": learn.MetricSpec(
            metric_fn= accuracy_metric,
            prediction_key="classes")}
  
  
    #Evaluate the model and print results
    eval_results = classifier.evaluate(
        input_fn=lambda: input_data_fn(TESTING_FILENAME,TESTING_BATCH_SIZE,1,TESTING_MIN_AFTER_DEQUEUE),
        metrics=testing_metrics,
        log_progress=False,
        steps=BATCHES_IN_TESTING_EPOCH-1)
     
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()