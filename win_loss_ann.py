'''
Created on Jul 2, 2017

@author: Samuel Ragusa
'''
import tensorflow as tf



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
 
 
#Making sure that the data pipeline works
with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
  
    for i in range(100):
        # Retrieve a single instance:
        example, label = sess.run([example_batch, label_batch])
        print(example, label)
        
    coord.request_stop()
    coord.join(threads)
    
