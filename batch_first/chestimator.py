import tensorflow as tf
import numpy as np

import batch_first as bf

from tensorflow.contrib import tensorrt as trt

from google.protobuf import text_format



def get_predictors(session, graphdef_filename, eval_output_tensor, move_output_stages_tensor_names, move_input_tensor_names, eval_input_tensor_names):
    """
    All ANN interactions should be called through C/C++ functions for speed (hopefully will be addressed soon)
    """
    with open(graphdef_filename, 'r') as f:
        txt = f.read()
        model_graph_def = text_format.Parse(txt, tf.GraphDef())

    desired_tensors = tf.import_graph_def(
        model_graph_def,
        return_elements=[eval_output_tensor] + move_output_stages_tensor_names + move_input_tensor_names + eval_input_tensor_names)

    eval_output = desired_tensors[0]
    move_outputs = desired_tensors[1:len(move_output_stages_tensor_names) + 1]
    move_inputs = desired_tensors[len(move_output_stages_tensor_names) + 1:-len(eval_input_tensor_names)]
    eval_inputs = desired_tensors[-len(eval_input_tensor_names):]

    with tf.device('/GPU:0'):
        with tf.control_dependencies([move_outputs[0]]):
            dummy_operation = tf.constant([0], dtype=tf.float32, name="dummy_const")

    def board_predictor(*inputs):
        return session.run(eval_output, dict(zip(eval_inputs, inputs)))

    def start_move_prediction(*board_inputs):
        handle = session.partial_run_setup([dummy_operation, move_outputs[1]], move_inputs)

        session.partial_run(handle, dummy_operation, dict(zip(move_inputs[:-3], board_inputs)))

        return lambda move_info: session.partial_run(handle,
                                                     move_outputs[1],
                                                     dict(zip(move_inputs[-3:], move_info)))

    return board_predictor, start_move_prediction


def get_inference_functions(graphdef_filename, session_gpu_memory=.4):
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=session_gpu_memory)))

    path_adder = lambda p, l : list(map(lambda st : "%s/%s"%(p, st), l))
    eval_output_tensor_name = "value_network/Squeeze:0"
    eval_input_tensor_names = [
        "piece_bbs:0",
        "color_occupied_bbs:0",
        "ep_squares:0",
        "castling_lookup_indices:0",
        "kings:0"]

    move_scoring_stages_names = ["Reshape_1", "requested_move_scores:0"]

    move_scoring_input_tensor_names = eval_input_tensor_names + [
        "from_square_placeholder:0", "move_filter_placeholder:0", "moves_per_board_placeholder:0"]

    eval_input_tensor_names = path_adder("value_network", eval_input_tensor_names)

    move_scoring_stages_names = path_adder("policy_network", move_scoring_stages_names)
    move_scoring_input_tensor_names = path_adder("policy_network", move_scoring_input_tensor_names)

    predictors = get_predictors(
        sess,
        graphdef_filename,
        eval_output_tensor_name, move_scoring_stages_names, move_scoring_input_tensor_names, eval_input_tensor_names)

    closer_fn = lambda: sess.close()

    return predictors[0], predictors[1], closer_fn


def get_board_data(data_format="NCHW"):
    piece_bbs = tf.placeholder(tf.int64, shape=[None, 1, 5], name="piece_bbs")
    color_occupied_bbs = tf.placeholder(tf.int64, shape=[None, 2, 1], name="color_occupied_bbs")
    ep_squares = tf.placeholder(tf.int32, shape=[None], name="ep_squares")
    castling_lookup_indices =tf.placeholder(tf.int32,shape=[None],name="castling_lookup_indices")
    kings = tf.placeholder(tf.int32, shape=[None, 2], name="kings")    #[,,,[white_king_square,black_king_square],,,]

    ep_lookup_array = np.stack([np.unpackbits(
            np.array([1 << sq],dtype=np.uint64).view(np.uint8)).reshape(8,8,1).astype(np.float32) if sq != 0 else np.zeros((8,8,1), dtype=np.float32) for sq in range(64)])

    ep_lookup_table = tf.constant(ep_lookup_array[...,::-1, :])

    kings_table = np.zeros([64,64,64,2],dtype=np.float32)
    aranged = np.arange(64)
    kings_table[aranged,:,aranged,0] = 1
    kings_table[:,aranged,aranged,1] = 1

    kings_table = kings_table.reshape([64,64,8,8,2])

    kings_lookup_table = tf.constant(kings_table, dtype=tf.float32)

    king_features = tf.gather_nd(kings_lookup_table, kings)


    castling_lookup_array = np.zeros((2**4, 8,8,2),dtype=np.bool_)
    possible_castling_square_array = np.array([bf.BB_A1, bf.BB_H1, bf.BB_A8, bf.BB_H8],dtype=np.uint64)

    castling_lookup_array[:, [0, 0, 7, 7], [0, 7, 0, 7], [0, 0, 1, 1]] = np.expand_dims(bf.POSSIBLE_CASTLING_RIGHTS.view(np.uint64), 1) & possible_castling_square_array != 0

    castling_lookup_table = tf.constant(castling_lookup_array,dtype=tf.float32)
    castling_features = tf.gather(castling_lookup_table, castling_lookup_indices)


    ep_bitboards = tf.gather(ep_lookup_table, ep_squares)

    color_specific_piece_info = tf.bitwise.bitwise_and(color_occupied_bbs, piece_bbs)

    reshaped_color_specific_info = tf.reshape(color_specific_piece_info, [-1, 10])


    the_bytes = tf.cast(tf.bitcast(reshaped_color_specific_info, tf.uint8), dtype=tf.int32)

    float_bool_masks = tf.constant(
        [np.unpackbits(num)[::-1].tolist() for num in np.arange(2 ** 8, dtype=np.uint8)],
        dtype=tf.float32)

    non_lookup_data = tf.gather(float_bool_masks, the_bytes)


    properly_arranged_non_lookup_data = tf.transpose(non_lookup_data, perm=[0, 2, 3, 1])

    full_data = tf.concat([
        ep_bitboards,
        king_features[...,0:1],
        properly_arranged_non_lookup_data[...,:5],
        castling_features[...,0:1],
        king_features[...,1:2],
        properly_arranged_non_lookup_data[...,5:],
        castling_features[...,1:2]], 3)

    # The below line of code will be used instead of the code above when inputs are eventually desired in that way
    # full_data = tf.concat([ep_bitboards, king_features, castling_features, properly_arranged_non_lookup_data], 3)

    if data_format == "NCHW":
        full_data = tf.transpose(full_data, [0,3,1,2], name="FOR_INPUT_MAPPING_transpose")

    return (piece_bbs, color_occupied_bbs, ep_squares, castling_lookup_indices, kings), full_data

