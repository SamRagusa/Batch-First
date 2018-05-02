import time
import tensorflow as tf
import numpy as np
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.estimator import _check_hooks_type
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import ops
from tensorflow.python.training import saver
from tensorflow.python.training import training
from tensorflow.python.client import timeline


def get_board_data(kings,queens,rooks,bishops,knights,pawns,castling_rights,ep_bitboards,occupied_w,occupied_b):#,occupied):
    non_castling_rooks = tf.bitwise.bitwise_xor(rooks, castling_rights)

    color_based_occupied = tf.stack([occupied_w, occupied_b], axis=1)
    info_about_pieces = tf.stack([kings, queens, non_castling_rooks, bishops, knights, pawns, castling_rights], axis=1)

    expanded_dim_color_occupied = tf.expand_dims(color_based_occupied, axis=2)
    expanded_dim_piece_info = tf.expand_dims(info_about_pieces, axis=1)

    color_specific_piece_info = tf.bitwise.bitwise_and(expanded_dim_color_occupied, expanded_dim_piece_info)

    reshaped_color_specific_info = tf.reshape(color_specific_piece_info, [-1, 14])

    expanded_dim_ep = tf.expand_dims(ep_bitboards, axis=1)

    the_ints = tf.concat([
        expanded_dim_ep,
        reshaped_color_specific_info], axis=1)

    the_bytes = tf.cast(tf.bitcast(the_ints, tf.uint8), dtype=tf.int32)

    float_bool_masks = tf.constant(
        [np.unpackbits(num).tolist() for num in np.arange(2 ** 8, dtype=np.uint8)],
        dtype=tf.float32)

    data = tf.gather(float_bool_masks, the_bytes)
    properly_arranged_data = tf.transpose(data, perm=[0, 2, 3, 1])

    return properly_arranged_data


def get_board_data_with_unoccupied(kings,queens,rooks,bishops,knights,pawns,castling_rights,ep_bitboards,occupied_w,occupied_b,occupied):
    not_occupied = tf.bitwise.invert(occupied)
    not_occupied_or_ep = tf.bitwise.bitwise_xor(not_occupied, ep_bitboards)
    non_castling_rooks = tf.bitwise.bitwise_xor(rooks, castling_rights)

    color_based_occupied = tf.stack([occupied_w, occupied_b], axis=1)
    info_about_pieces = tf.stack([kings, queens, non_castling_rooks, bishops, knights, pawns, castling_rights], axis=1)

    expanded_dim_color_occupied = tf.expand_dims(color_based_occupied, axis=2)
    expanded_dim_piece_info = tf.expand_dims(info_about_pieces, axis=1)

    color_specific_piece_info = tf.bitwise.bitwise_and(expanded_dim_color_occupied, expanded_dim_piece_info)

    reshaped_color_specific_info = tf.reshape(color_specific_piece_info, [-1, 14])

    expanded_dim_ep = tf.expand_dims(ep_bitboards, axis=1)
    expanded_dim_not_occupied_or_ep = tf.expand_dims(not_occupied_or_ep, axis=1)

    the_ints = tf.concat([expanded_dim_not_occupied_or_ep,
                          expanded_dim_ep,
                          reshaped_color_specific_info], axis=1)

    the_bytes = tf.cast(tf.bitcast(the_ints, tf.uint8), dtype=tf.int32)

    float_bool_masks = tf.constant(
        [np.unpackbits(num).tolist() for num in np.arange(2 ** 8, dtype=np.uint8)],
        dtype=tf.float32)

    data = tf.gather(float_bool_masks, the_bytes)

    properly_arranged_data = tf.transpose(data, perm=[0, 2, 3, 1])

    return properly_arranged_data







class ChEstimator(Estimator):

    def create_predictor(self,
                         predict_keys=None,
                         hooks=None,
                         checkpoint_path=None,
                         use_full_gpu=False):

        run_metadata = tf.RunMetadata()
        hooks = _check_hooks_type(hooks)

        # Check that model has been trained.
        if not checkpoint_path:
            checkpoint_path = saver.latest_checkpoint(self._model_dir)
        if not checkpoint_path:
            raise ValueError('Could not find trained model in model_dir: {}.'.format(
                self._model_dir))

        with ops.Graph().as_default():

            kings = tf.placeholder(tf.int64, shape=[None], name="kings_placeholder")
            queens = tf.placeholder(tf.int64, shape=[None], name="queens_placeholder")
            rooks = tf.placeholder(tf.int64, shape=[None], name="rooks_placeholder")
            bishops = tf.placeholder(tf.int64, shape=[None], name="bishops_placeholder")
            knights = tf.placeholder(tf.int64, shape=[None], name="knights_placeholder")
            pawns = tf.placeholder(tf.int64, shape=[None], name="pawns_placeholder")
            castling_rights = tf.placeholder(tf.int64, shape=[None], name="castling_rights_placeholder")
            ep_bitboards = tf.placeholder(tf.int64, shape=[None], name="ep_bitboards_placeholder")
            occupied_w = tf.placeholder(tf.int64, shape=[None], name="occupied_w_placeholder")
            occupied_b = tf.placeholder(tf.int64, shape=[None], name="occupied_b_placeholder")
            # occupied = tf.placeholder(tf.int64, shape=[None], name="occupied_placeholder")


            properly_arranged_data = get_board_data(kings, queens, rooks, bishops, knights, pawns, castling_rights, ep_bitboards, occupied_w, occupied_b)#, occupied)

            estimator_spec = self._call_model_fn({"feature": properly_arranged_data}, None,
                                                 model_fn_lib.ModeKeys.PREDICT, None)
            predictions = self._extract_keys(estimator_spec.predictions, predict_keys)[predict_keys]

            squeezed_predictions = tf.squeeze(predictions, axis=1)

            mon_sess = training.MonitoredSession(
                session_creator=training.ChiefSessionCreator(
                    checkpoint_filename_with_path=checkpoint_path,
                    scaffold=estimator_spec.scaffold,
                    # config=self._session_config),
                    config=self._session_config if use_full_gpu else tf.ConfigProto(
                        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.45))),
                hooks=hooks)

            def predictor(bb_array):
                if mon_sess.should_stop():
                    raise StopIteration

                return mon_sess.run(squeezed_predictions,
                                    {kings: bb_array[0],
                                     queens: bb_array[1],
                                     rooks: bb_array[2],
                                     bishops: bb_array[3],
                                     knights: bb_array[4],
                                     pawns: bb_array[5],
                                     castling_rights: bb_array[6],
                                     ep_bitboards: bb_array[7],
                                     occupied_w: bb_array[8],
                                     occupied_b: bb_array[9],
                                     })#occupied: bb_array[10]}, )
                # options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                # run_metadata=run_metadata)



            def finish():
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('timeline_eval.json', 'w') as f:
                    f.write(chrome_trace)
                mon_sess.close()

            return predictor, finish




class MoveChEstimator(Estimator):

    def create_move_predictor(self,
                              predict_keys=None,
                              hooks=None,
                              checkpoint_path=None,
                              max_moves_for_a_board=100,
                              max_batch_size=50000):


        run_metadata = tf.RunMetadata()
        hooks = _check_hooks_type(hooks)

        # Check that model has been trained.
        if not checkpoint_path:
            checkpoint_path = saver.latest_checkpoint(self._model_dir)
        if not checkpoint_path:
            raise ValueError('Could not find trained model in model_dir: {}.'.format(
                self._model_dir))

        kings = tf.placeholder(tf.int64, shape=[None], name="kings_placeholder")
        queens = tf.placeholder(tf.int64, shape=[None], name="queens_placeholder")
        rooks = tf.placeholder(tf.int64, shape=[None], name="rooks_placeholder")
        bishops = tf.placeholder(tf.int64, shape=[None], name="bishops_placeholder")
        knights = tf.placeholder(tf.int64, shape=[None], name="knights_placeholder")
        pawns = tf.placeholder(tf.int64, shape=[None], name="pawns_placeholder")
        castling_rights = tf.placeholder(tf.int64, shape=[None], name="castling_rights_placeholder")
        ep_bitboards = tf.placeholder(tf.int64, shape=[None],
                                      name="ep_bitboards_placeholder")  # Should double check this isn't interpreting uint8 values as negative int32 values
        occupied_w = tf.placeholder(tf.int64, shape=[None], name="occupied_w_placeholder")
        occupied_b = tf.placeholder(tf.int64, shape=[None], name="occupied_b_placeholder")
        # occupied = tf.placeholder(tf.int64, shape=[None], name="occupied_placeholder")

        move_nums = tf.placeholder(tf.int32, shape=[None], name="move_nums_placeholder")
        moves_per_board = tf.placeholder(tf.uint8, shape=[None], name="moves_per_board_placeholder")

        # These will be used soon
        # moves_from_square = tf.placeholder(tf.uint8, shape=[None], name="moves_from_square_placeholder")
        # moves_to_square = tf.placeholder(tf.uint8, shape=[None], name="moves_to_square_placeholder")
        # moves = tf.placeholder(tf.int32, shape=[None, 2], name="moves_placeholder")








        properly_arranged_data = get_board_data(kings, queens, rooks, bishops, knights, pawns, castling_rights,
                                                ep_bitboards, occupied_w, occupied_b)
        # properly_arranged_data = get_board_data_with_unoccupied(kings, queens, rooks, bishops, knights, pawns, castling_rights,
        #                                                         ep_bitboards, occupied_w, occupied_b, occupied)

        estimator_spec = self._call_model_fn({"data": properly_arranged_data}, None, model_fn_lib.ModeKeys.PREDICT,
                                             None)

        predictions = self._extract_keys(estimator_spec.predictions, predict_keys)[predict_keys]

        board_index_repeated_array = tf.transpose(
            tf.reshape(
                tf.tile(
                    tf.range(max_moves_for_a_board),
                    [max_batch_size]),
                [max_batch_size, max_moves_for_a_board]),
            [1, 0])




        board_indices_for_moves = tf.boolean_mask(board_index_repeated_array, tf.sequence_mask(tf.cast(moves_per_board,tf.int32)))



        the_moves = tf.stack([board_indices_for_moves, move_nums], axis=-1)


        legal_move_scores = tf.gather_nd(predictions, the_moves)

        mon_sess = training.MonitoredSession(
            session_creator=training.ChiefSessionCreator(
                checkpoint_filename_with_path=checkpoint_path,
                scaffold=estimator_spec.scaffold,
                # config=self._session_config),
                config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.45))),
            hooks=hooks)


        def predictor(bb_array, the_moves, num_moves_per_board):
            if mon_sess.should_stop():
                raise StopIteration

            return mon_sess.run(legal_move_scores,
                                {kings: bb_array[0],
                                 queens: bb_array[1],
                                 rooks: bb_array[2],
                                 bishops: bb_array[3],
                                 knights: bb_array[4],
                                 pawns: bb_array[5],
                                 castling_rights: bb_array[6],
                                 ep_bitboards: bb_array[7],
                                 occupied_w: bb_array[8],
                                 occupied_b: bb_array[9],
                                 # occupied: bb_array[10],
                                 move_nums: the_moves,
                                 moves_per_board : num_moves_per_board})

        def finish():
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()

            with open('timeline_move_scoring.json', 'w') as f:
                f.write(chrome_trace)

            mon_sess.close()

        return predictor, finish

