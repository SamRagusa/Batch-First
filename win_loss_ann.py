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

    
