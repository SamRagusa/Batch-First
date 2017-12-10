'''
Created on Jul 2, 2017

@author: SamRagusa
'''

import tensorflow as tf

import ann_creation_helper_for_commit as ann_h

from functools import reduce



tf.logging.set_verbosity(tf.logging.INFO)




# Learn how to use this parameter, and write this method to generate a run
# based on the information given in the parameter
def main(unused_param):
    """
    Set up the data pipelines, create the computational graph, train the model, and evaluate the results.
    """
    SAVE_MODEL_DIR = "/home/sam/Desktop/tmp/pre_commit_test"
    TRAINING_FILENAMES = ["/srv/databases/chess_engine/full_3/shuffled_training_set.txt"]
    VALIDATION_FILENAMES = ["/srv/databases/chess_engine/full_5/scoring_validation_set.tfrecords"]
    TRAIN_OP_SUMMARIES = ["gradient_norm", "gradients"]
    NUM_OUTPUTS = 1
    DENSE_SHAPE = [1024,512,256]
    OPTIMIZER = 'Adam'
    TRAINING_MIN_AFTER_DEQUEUE = 20000
    VALIDATION_MIN_AFTER_DEQUEUE = 10000
    TRAINING_BATCH_SIZE = 250
    EQUALITY_SCALAR_MULT = 1
    VALIDATION_BATCH_SIZE = 2000
    NUM_TRAINING_EPOCHS = 500
    LOG_ITERATION_INTERVAL = 5000
    LEARNING_RATE = .00001



    INCEPTION_MODULES = [
        [
            [[32, 2], [50, 2], [64, 2]],
            [[32, 3], [64, 2]],
            [[5, 4]]],  # 133 5x5 feature map outputs
        [
            [[64, 1]],
            [[32, 1], [16, 2]],
            [[32, 1], [16, 3]]]]  # output of 2000  neurons



    BATCHES_IN_TRAINING_EPOCH = 758545383  // (TRAINING_BATCH_SIZE)
    #reduce(
        # lambda x, y: x + y,
        # [ann_h.line_counter(filename) for filename in TRAINING_FILENAMES]) // (TRAINING_BATCH_SIZE * 10)
    BATCHES_IN_VALIDATION_EPOCH = 1000000// VALIDATION_BATCH_SIZE
    #84289074 // VALIDATION_BATCH_SIZE
    #ann_h.line_counter(VALIDATION_FILENAMES[0]) // VALIDATION_BATCH_SIZE


    learning_decay_function = lambda gs  : tf.train.exponential_decay(LEARNING_RATE,
                                                                      global_step=gs,
                                                                      decay_steps=BATCHES_IN_TRAINING_EPOCH//60,
                                                                      decay_rate=0.9,
                                                                      staircase=True)

    print(BATCHES_IN_TRAINING_EPOCH)
    print(BATCHES_IN_VALIDATION_EPOCH)


    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=ann_h.cnn_model_fn,
        model_dir=SAVE_MODEL_DIR,
        config=tf.estimator.RunConfig().replace(
            save_checkpoints_steps=LOG_ITERATION_INTERVAL,
            save_summary_steps=LOG_ITERATION_INTERVAL),
            # session_config=tf.ConfigProto(log_device_placement=True)),
        params={
            'dense_shape': DENSE_SHAPE,
            'optimizer': OPTIMIZER,
            'num_outputs': NUM_OUTPUTS,
            'log_interval': LOG_ITERATION_INTERVAL,
            'model_dir': SAVE_MODEL_DIR,
            'inception_modules' : INCEPTION_MODULES,
            "learning_rate": LEARNING_RATE,
            "train_summaries": TRAIN_OP_SUMMARIES,
            'learning_decay_function' : learning_decay_function,
            'equality_scalar': EQUALITY_SCALAR_MULT})

    validation_hook = ann_h.ValidationRunHook(
        step_increment=BATCHES_IN_TRAINING_EPOCH//6000,
        estimator=classifier,
        filenames=VALIDATION_FILENAMES,
        batch_size=VALIDATION_BATCH_SIZE,
        min_after_dequeue=VALIDATION_MIN_AFTER_DEQUEUE,
        temp_num_steps_in_epoch=500)#BATCHES_IN_VALIDATION_EPOCH)

    classifier.train(
        # Could likely just create pipeline in similar way to how I did in ValidationRunHook,
        # eliminating the need for an input function
        input_fn=lambda: ann_h.input_data_fn(
            TRAINING_FILENAMES,
            TRAINING_BATCH_SIZE,
            NUM_TRAINING_EPOCHS,
            TRAINING_MIN_AFTER_DEQUEUE,
            True),
        hooks=[validation_hook])


    # Export the model for serving
    classifier.export_savedmodel(
        SAVE_MODEL_DIR,
        serving_input_receiver_fn=ann_h.serving_input_receiver_fn)






if __name__ == "__main__":
    tf.app.run()
