'''
Created on Jul 2, 2017

@author: Samuel Ragusa
'''
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib



tf.logging.set_verbosity(tf.logging.INFO)


if __name__ == "__main__":
    tf.app.run()


sess = tf.InteractiveSession()


def data_pipeline(filenames, batch_size, num_epochs=None, min_after_dequeue=10000):
    """
    Creates a pipeline for the data contained within the given files.  
    It does this using TensorFlow queues.
    
    @return: A tuple in which the first element is a graph operation
    which gets a random batch of the data, and the second element is
    a graph operation to get a batch of the labels corresponding
    to the data gotten by the first element of the tuple.
    """
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
    
    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)
    
    row = tf.decode_csv(value, record_defaults=[[0.0] for _ in range(66)])
    
    #change the row[:len(row)-2] type stuff to be more pythonic
    example_op, label_op  = tf.stack(row[:len(row)-2]), tf.stack(row[len(row)-2:])
    
    capacity = min_after_dequeue + 3 * batch_size 
    
    example_batch, label_batch = tf.train.shuffle_batch(
        [example_op, label_op], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch


FILENAMES = ["small_dataset.csv"]
BATCH_SIZE = 100
MIN_AFTER_DEQUEUE = 500


example_batch, label_batch = data_pipeline(FILENAMES,BATCH_SIZE, min_after_dequeue=MIN_AFTER_DEQUEUE)

   

def cnn_model_fn(features, labels, mode):
    """
    Model function for the CNN.
    """
    
    def build_inception_module(the_input, module):
        """
        Module of the format:
        [[[filters1_1,kernal_size1_1],... , [filters1_M,kernal_size1_M]],... ,
            [filtersN_1,kernal_sizeN_1],... , [filtersN_P,kernal_sizeN_P]]
            
        (MAYBE IMPLEMENT) A set of [filtersJ_I, kernal_sizeJ_I] can be replaced by None for passing of input to concatination
        
        NOTES:
        1) Add naming for use with TensorBoard
        2)test to see if this works
        """
        
        path_outputs = [None for _ in range(len(module))]  ##See if I can do [None]*len(module) instead
        
        cur_input = None
        for j, path in enumerate(module):
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
                    activation=tf.nn.relu)
        
        path_outputs[-1] = cur_input
        return tf.nn.relu(tf.concat([temp_input for temp_input in path_outputs],3))
    
    
    def build_fully_connected_layers(the_input, shape, dropout_rates=None):
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
        
        if dropout_rates == None:
            dropout_rates = [0]*len(shape)
        elif not isinstance(dropout_rates, list):
            if dropout_rates >= 0 and dropout_rates < 1:
                dropout_rates = [dropout_rates]*len(shape)
            else:
                #PRINT THE ERROR
                pass
        else:
            if len(dropout_rates) != len(shape):
                #PRINT THE ERROR
                pass       
        
        for size, dropout in zip(shape, dropout_rates):
            temp_layer_dense = tf.layers.dense(inputs=the_input, units=size, activation=tf.nn.relu)
            if dropout != 0:
                the_input =  tf.layers.dropout(inputs=temp_layer_dense, rate=dropout, training=mode == learn.ModeKeys.TRAIN)
            else:
                the_input = temp_layer_dense
        
        return the_input
    
    
    input_layer = tf.reshape(features, [-1,8,8,1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
            activation=tf.nn.relu)
    
    conv2_flat = tf.reshape(conv2, [-1, 8 * 8 * 64])
    
    dense = tf.layers.dense(inputs=conv2_flat, units=1024, activation=tf.nn.relu)
    
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
    
    logits = tf.layers.dense(inputs=dropout, units=2)
    
    loss = None
    train_op = None
    
    # Calculate loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)
        
    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD")
        
    # Generate Predictions
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(
        logits, name="softmax_tensor")
    }
        
    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)
    
