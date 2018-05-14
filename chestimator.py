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

from numba_board import generate_move_to_enumeration_dict


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

    return piece_bbs, color_occupied_bbs, ep_squares, full_data






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

            piece_bbs, color_occupied_bbs, ep_squares, for_evaluation = new_get_board_data()

            estimator_spec = self._call_model_fn({"feature": for_evaluation}, None,
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


            def predictor(pieces, occupied_bbs, ep_square_numbers):
                if mon_sess.should_stop():
                    raise StopIteration

                return mon_sess.run(squeezed_predictions,
                                    {piece_bbs: pieces,
                                     color_occupied_bbs: occupied_bbs,
                                     ep_squares: ep_square_numbers,
                                     },)
                                    # options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                    # run_metadata=run_metadata)


            def finish():
            #     fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            #     chrome_trace = fetched_timeline.generate_chrome_trace_format()
            #     with open('timeline_eval.json', 'w') as f:
            #         f.write(chrome_trace)
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


        moves_per_board = tf.placeholder(tf.uint8, shape=[None], name="moves_per_board_placeholder")
        moves = tf.placeholder(tf.uint8, shape=[None, 2], name="move_placeholder")

        piece_bbs, color_occupied_bbs, ep_squares, for_evaluation = new_get_board_data()


        move_to_index_array = np.zeros(shape=[64, 64], dtype=np.int32)
        for key, value in generate_move_to_enumeration_dict().items():
            move_to_index_array[key[0], key[1]] = value

        move_to_index_tensor = tf.constant(move_to_index_array, shape=[64, 64])


        estimator_spec = self._call_model_fn({"data": for_evaluation}, None, model_fn_lib.ModeKeys.PREDICT,
                                             None)

        predictions = self._extract_keys(estimator_spec.predictions, predict_keys)[predict_keys]




        board_index_repeated_array = tf.transpose(
            tf.reshape(
                tf.tile(
                    tf.range(max_moves_for_a_board),
                    [max_batch_size]),
                [max_batch_size, max_moves_for_a_board]),
            [1, 0])

        board_indices_for_moves = tf.boolean_mask(board_index_repeated_array,
                                                  tf.sequence_mask(tf.cast(moves_per_board, tf.int32)))

        move_nums = tf.gather_nd(move_to_index_tensor, tf.cast(moves, tf.int32))

        the_moves = tf.stack([board_indices_for_moves, move_nums], axis=-1)

        legal_move_scores = tf.gather_nd(predictions, the_moves)

        mon_sess = training.MonitoredSession(
            session_creator=training.ChiefSessionCreator(
                checkpoint_filename_with_path=checkpoint_path,
                scaffold=estimator_spec.scaffold,
                # config=self._session_config),
                config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.45))),
            hooks=hooks)

        def predictor(pieces, occupied_bbs, ep_square_numbers, the_moves, num_moves_per_board):
            if mon_sess.should_stop():
                raise StopIteration

            return mon_sess.run(legal_move_scores,
                                {piece_bbs: pieces,
                                 color_occupied_bbs: occupied_bbs,
                                 ep_squares: ep_square_numbers,
                                 moves: the_moves,
                                 moves_per_board: num_moves_per_board,
                                 }, )
            # options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            # run_metadata=run_metadata)



        def finish():
            # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            # chrome_trace = fetched_timeline.generate_chrome_trace_format()
            #
            # with open('timeline_move_scoring.json', 'w') as f:
            #     f.write(chrome_trace)

            mon_sess.close()

        return predictor, finish
