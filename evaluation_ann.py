'''
Created on Jul 2, 2017

@author: SamRagusa
'''
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python import training

import ann_creation_helper as ann_h

from functools import reduce

tf.logging.set_verbosity(tf.logging.INFO)




# Figure out how to use this parameter, and write this method to generate a run
# based on the information given in the parameter
def main(unused_param):
    """
    Set up the data pipelines, create the computational graph, train the model, and evaluate the results.
    """

    # Hopefully set these constants up with something like flags to better be able
    # to run modified versions of the model in the future (when tuning the model)
    SAVE_MODEL_DIR = "/home/sam/Desktop/tmp/pre_commit_test_999"
    TRAINING_FILENAMES = ["deep_pink_tester.txt"]
    VALIDATION_FILENAMES = ["deep_pink_tester_leq_5_fics2015.txt"]
    TESTING_FILENAMES = ["new_test_data.csv"]
    TRAIN_OP_SUMMARIES = ["learning_rate", "loss", "gradient_norm", "gradients"]
    NUM_OUTPUTS = 1
    DENSE_SHAPE = [2048, 1024]  # NEED TO PICK THESE PROPERLY
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
    LOG_ITERATION_INTERVAL = 2500
    LEARNING_RATE = .0001  # NEED TO PICK THIS PROPERLY
    INPUT_CONV_FILTERS = 400
    INPUT_CONV_KERNAL = [2, 2]
    INCEPTION_MODULES = [
        [
            [[75, 2], [100, 2], [125, 2]],
            [[100, 2], [125, 3]],  # Very likely want the 3x3 convolution to come before the 2x2
            [[125, 4]]],
        [
            [[75, 1]],
            [[100, 1], [50, 2]],
            [[100, 1], [60, 3]]]]    # output of 3215 neurons

    BATCHES_IN_TRAINING_EPOCH = 10000  # reduce(
    # lambda x, y: x + y,
    # [line_counter(filename) for filename in TRAINING_FILENAMES]) // TRAINING_BATCH_SIZE
    BATCHES_IN_VALIDATION_EPOCH = ann_h.line_counter(VALIDATION_FILENAMES[0]) // VALIDATION_BATCH_SIZE
    # BATCHES_IN_TESTING_EPOCH = line_counter(TESTING_FILENAMES)//TESTING_BATCH_SIZE

    # print(BATCHES_IN_TRAINING_EPOCH)
    # print(BATCHES_IN_VALIDATION_EPOCH)

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
            'dense_dropout': DENSE_DROPOUT,
            'optimizer': OPTIMIZER,
            'num_outputs': NUM_OUTPUTS,
            'log_interval': LOG_ITERATION_INTERVAL,
            'model_dir': SAVE_MODEL_DIR,
            'inception_modules' : INCEPTION_MODULES,
            "learning_rate": LEARNING_RATE,
            "train_summaries": TRAIN_OP_SUMMARIES,
            'loss_fn': LOSS_FN,
            'input_conv_filters': INPUT_CONV_FILTERS,
            'input_conv_kernal': INPUT_CONV_KERNAL,
            'init_bias_value': INITIAL_BIAS_VALUE,
            'equality_scalar': EQUALITY_SCALAR_MULT})

    validation_hook = ann_h.ValidationRunHook(
        step_increment=BATCHES_IN_TRAINING_EPOCH,
        estimator=classifier,
        filenames=VALIDATION_FILENAMES,
        batch_size=VALIDATION_BATCH_SIZE,
        min_after_dequeue=VALIDATION_MIN_AFTER_DEQUEUE,
        temp_num_steps_in_epoch=BATCHES_IN_VALIDATION_EPOCH)

    classifier.train(
        # Could likely just create pipeline in similar way to how I did in ValidationRunHook,
        # eliminating the need for an input function
        input_fn=lambda: ann_h.input_data_fn(
            TRAINING_FILENAMES,
            TRAINING_BATCH_SIZE,
            NUM_TRAINING_EPOCHS,
            TRAINING_MIN_AFTER_DEQUEUE,
            True),
        # lambda: data_pipeline([TRAINING_FILENAME], TRAINING_BATCH_SIZE, num_epochs, min_after_dequeue))
        hooks=[validation_hook])#, max_steps=1)


    # Export the model for serving
    classifier.export_savedmodel(
        SAVE_MODEL_DIR,
        serving_input_receiver_fn=ann_h.serving_input_receiver_fn)

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
