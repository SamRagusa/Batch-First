import tensorflow as tf
import numpy as np

from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

#If the graph hasn't been optimized with TensorRT, this import can be commented out.
from tensorflow.contrib import tensorrt as trt




def get_predictor_from_graphdef(session, graphdef_filename, output_tensor, input_tensors, name_prefix=None, is_binary=True):
    with gfile.FastGFile(graphdef_filename, 'rb' if is_binary else 'r') as f:
        model_graph_def = tf.GraphDef()
        model_graph_def.ParseFromString(compat.as_bytes(f.read()))

        desired_tensors = tf.import_graph_def(
            model_graph_def,
            return_elements=[output_tensor] + input_tensors,
            name=name_prefix)

        def test_predictor(*inputs):
            return session.run(desired_tensors[0], dict(zip(desired_tensors[1:], inputs)))

        return test_predictor



def get_inference_functions(eval_graphdef_file, move_graphdef_file):
    """
    The code relevant to move scoring has been commented out as it's not ready yet.
    """

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.3)))

    eval_output_tensor_name = "add:0"#"logit_layer/MatMul:0"#
    eval_input_tensor_names = [
        "piece_bbs:0",
        "color_occupied_bbs:0",
        "ep_squares:0"]

    move_scoring_output_tensor_name = "GatherNd_1:0"
    move_scoring_input_tensor_names = [
        "piece_bbs:0",
        "color_occupied_bbs:0",
        "ep_squares:0",
        "move_placeholder:0",
        "moves_per_board_placeholder:0"]


    evaluation_predictor = get_predictor_from_graphdef(
        sess,
        eval_graphdef_file,
        eval_output_tensor_name,
        eval_input_tensor_names,
        "board_eval",
        True)

    move_predictor = get_predictor_from_graphdef(
        sess,
        move_graphdef_file,
        move_scoring_output_tensor_name,
        move_scoring_input_tensor_names,
        "move_scoring",
        True)


    closer_fn = lambda: sess.close()

    return evaluation_predictor, move_predictor, closer_fn






def new_get_board_data():
    piece_bbs = tf.placeholder(tf.int64, shape=[None, 1, 7], name="piece_bbs")
    color_occupied_bbs = tf.placeholder(tf.int64, shape=[None, 2, 1], name="color_occupied_bbs")
    ep_squares = tf.placeholder(tf.int32, shape=[None], name="ep_squares")

    ep_lookup_table = tf.constant(
        np.stack([np.unpackbits(
            np.array([1 << sq],dtype=np.uint64).view(np.uint8)).reshape(8,8,1).astype(np.float32) if sq != 0 else np.zeros((8,8,1), dtype=np.float32) for sq in range(64)]))


    ep_bitboards = tf.gather(ep_lookup_table, ep_squares)

    color_specific_piece_info = tf.bitwise.bitwise_and(color_occupied_bbs, piece_bbs)

    reshaped_color_specific_info = tf.reshape(color_specific_piece_info, [-1, 14])


    the_bytes = tf.cast(tf.bitcast(reshaped_color_specific_info, tf.uint8), dtype=tf.int32)

    float_bool_masks = tf.constant(
        [np.unpackbits(num).tolist() for num in np.arange(2 ** 8, dtype=np.uint8)],
        dtype=tf.float32)


    non_ep_data = tf.gather(float_bool_masks, the_bytes)


    properly_arranged_non_ep_data = tf.transpose(non_ep_data, perm=[0, 2, 3, 1])

    full_data = tf.concat([ep_bitboards, properly_arranged_non_ep_data], 3)

    return (piece_bbs, color_occupied_bbs, ep_squares), full_data




