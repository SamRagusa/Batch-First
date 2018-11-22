import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.platform import gfile

from tensorflow.contrib import tensorrt as trt



def get_predictors(session, graphdef_filename, eval_output_tensor, move_output_stages_tensor_names, move_input_tensor_names, eval_input_tensor_names):
    """
    All ANN interactions should be called through C/C++ functions for speed (hopefully will be addressed soon)
    """
    if graphdef_filename[-3:] == ".pb":
        with gfile.FastGFile(graphdef_filename, 'rb') as f:
            model_graph_def = tf.GraphDef()
            model_graph_def.ParseFromString(f.read())
    else:
        with open(graphdef_filename, 'r') as f:
            txt = f.read()
            model_graph_def = text_format.Parse(txt, tf.GraphDef())

    desired_tensors = tf.import_graph_def(
        model_graph_def,
        return_elements=[eval_output_tensor] + move_output_stages_tensor_names + move_input_tensor_names + eval_input_tensor_names,
        name="")

    eval_output = desired_tensors[0]
    move_outputs = desired_tensors[1:len(move_output_stages_tensor_names) + 1]
    move_inputs = desired_tensors[len(move_output_stages_tensor_names) + 1:-len(eval_input_tensor_names)]
    eval_inputs = desired_tensors[-len(eval_input_tensor_names):]

    with tf.device('/GPU:0'):
        with tf.control_dependencies([move_outputs[0]]):
            dummy_operation = tf.constant([0], dtype=tf.float32, name="dummy_const")


    board_predictor = session.make_callable(eval_output_tensor, eval_input_tensor_names)

    def start_move_prediction(*board_inputs):
        handle = session.partial_run_setup([dummy_operation, move_outputs[1]], eval_inputs + move_inputs)
        session.partial_run(handle, dummy_operation, dict(zip(eval_inputs, board_inputs)))

        return lambda move_info: session.partial_run(handle,
                                                     move_outputs[1],
                                                     dict(zip(move_inputs[-3:], move_info)))

    return board_predictor, start_move_prediction


def get_inference_functions(graphdef_filename, session_gpu_memory=.4):
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=session_gpu_memory)))

    path_adder = lambda p, l : list(map(lambda st : "%s/%s"%(p, st), l))
    eval_output_tensor_name = "value_network/Squeeze:0"
    eval_input_tensor_names = ["piece_filters:0", "occupied_bbs:0"]

    move_scoring_stages_names = ["Reshape", "requested_move_scores:0"]

    move_scoring_input_tensor_names = ["from_square_placeholder:0", "move_filter_placeholder:0", "moves_per_board_placeholder:0"]

    eval_input_tensor_names = path_adder("input_parser", eval_input_tensor_names)

    move_scoring_stages_names = path_adder("policy_network", move_scoring_stages_names)
    move_scoring_input_tensor_names = path_adder("policy_network", move_scoring_input_tensor_names)

    predictors = get_predictors(
        sess,
        graphdef_filename,
        eval_output_tensor_name, move_scoring_stages_names, move_scoring_input_tensor_names, eval_input_tensor_names)

    closer_fn = lambda: sess.close()

    return predictors[0], predictors[1], closer_fn



