import tensorflow as tf
import numpy as np

from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import batch_first as bf

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


def get_move_predictor(session, graphdef_filename, output_stages_tensor_names, input_tensor_names, name_prefix=None, is_binary=True):
    with gfile.FastGFile(graphdef_filename, 'rb' if is_binary else 'r') as f:
        model_graph_def = tf.GraphDef()
        model_graph_def.ParseFromString(compat.as_bytes(f.read()))

        desired_tensors = tf.import_graph_def(
            model_graph_def,
            return_elements=output_stages_tensor_names + input_tensor_names,
            name=name_prefix)

    def start_move_prediction(*board_inputs):
        handle = session.partial_run_setup(desired_tensors[:2], desired_tensors[2:])


        # If I'm not mistaken the workaround mentioned below will be much faster when moved to TensorFlow 1.10,
        # but either way, all ANN interactions should be through C/C++ functions for Numba to call (long term)

        #############THIS IS NOT RETURNING None, INSTEAD ITS RETURNING (all logits) AND THEYRE JUST BEING IGNORED, BUT THIS IS VERY CRUCIAL FOR SPEED AND A LAUGHABLY SLOW SOLTUION###############
        session.partial_run(handle, desired_tensors[0], dict(zip(desired_tensors[2: 7], board_inputs)))

        return lambda move_info: session.partial_run(handle,
                                                     desired_tensors[1],
                                                     dict(zip(desired_tensors[7:], move_info)))

    return start_move_prediction


def get_inference_functions(eval_graphdef_file, move_graphdef_file, session_gpu_memory=.4):
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=session_gpu_memory)))

    eval_output_tensor_name = "Squeeze:0"
    eval_input_tensor_names = [
        "piece_bbs:0",
        "color_occupied_bbs:0",
        "ep_squares:0",
        "castling_lookup_indices:0",
        "kings:0"]

    move_scoring_stages_names = ["logit_layer/Conv2D:0", "GatherNd_1:0"]
    move_scoring_input_tensor_names = eval_input_tensor_names + ["move_placeholder:0", "moves_per_board_placeholder:0"]

    evaluation_predictor = get_predictor_from_graphdef(
        sess,
        eval_graphdef_file,
        eval_output_tensor_name,
        eval_input_tensor_names,
        "board_eval")

    if move_graphdef_file is None:
        move_predictor = None
    else:
        move_predictor = get_move_predictor(
            sess,
            move_graphdef_file,
            move_scoring_stages_names,
            move_scoring_input_tensor_names,
            "move_scoring")

    closer_fn = lambda: sess.close()

    return evaluation_predictor, move_predictor, closer_fn


def get_board_data(data_format="NCHW"):
    """
    NOTES:
    1) Verify that accepting the uint8's as int32 is the best way to do this, casting is relatively fast so doing that
    wouldn't be a huge deal.
    """
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
        full_data = tf.transpose(full_data, [0,3,1,2])

    return (piece_bbs, color_occupied_bbs, ep_squares, castling_lookup_indices, kings), full_data

