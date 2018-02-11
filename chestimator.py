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






class ChEstimator(Estimator):

    def create_predictor(self,
                        predict_keys=None,
                        hooks=None,
                        checkpoint_path=None):


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
            ep_square = tf.placeholder(tf.int32, shape=[None], name="ep_square_placeholder")  # Should double check this isn't interpreting uint8 values as negative int32 values
            occupied_w = tf.placeholder(tf.int64, shape=[None], name="occupied_w_placeholder")
            occupied_b = tf.placeholder(tf.int64, shape=[None], name="occupied_b_placeholder")
            occupied = tf.placeholder(tf.int64, shape=[None], name="occupied_placeholder")
            turn_white = tf.placeholder(tf.bool, shape=[None], name="turn_placeholder")

            #This is overlycomplicated in regards to typing, but doesn't effect runtime speed
            bb_squares = tf.constant(np.array([np.uint64(1 << sq) for sq in range(64)]+ [0 for _ in range(64, 256)],
                                              dtype=np.uint64).astype(np.int64), tf.int64)

            ep_bitboards = tf.gather(bb_squares, ep_square)

            first_players_occupied = tf.where(turn_white, occupied_w, occupied_b)
            second_players_occupied = tf.where(turn_white, occupied_b, occupied_w)

            # This could likely be done in a more simple way
            the_ints = tf.stack([
                tf.bitwise.invert(occupied),
                ep_bitboards,
                tf.bitwise.bitwise_and(first_players_occupied, kings),  # Likely can do without AND operation
                tf.bitwise.bitwise_and(first_players_occupied, queens),
                tf.bitwise.bitwise_and(first_players_occupied, rooks),
                tf.bitwise.bitwise_and(first_players_occupied, bishops),
                tf.bitwise.bitwise_and(first_players_occupied, knights),
                tf.bitwise.bitwise_and(first_players_occupied, pawns),
                tf.bitwise.bitwise_and(first_players_occupied, castling_rights),# Very likely should do this differently (to avoid 8 indices in tf.gather and instead use 1)
                tf.bitwise.bitwise_and(second_players_occupied, kings),  # Likely can do without AND operation
                tf.bitwise.bitwise_and(second_players_occupied, queens),
                tf.bitwise.bitwise_and(second_players_occupied, rooks),
                tf.bitwise.bitwise_and(second_players_occupied, bishops),
                tf.bitwise.bitwise_and(second_players_occupied, knights),
                tf.bitwise.bitwise_and(second_players_occupied, pawns),
                tf.bitwise.bitwise_and(second_players_occupied, castling_rights),# Very likely should do this differently (to avoid 8 indices in tf.gather and instead use 1)
            ], axis=1)

            the_bytes = tf.cast(tf.bitcast(the_ints, tf.uint8), dtype=tf.int32)

            #I haven't switched this to the dynamic partition/stich method used for move scoring because for some reason
            #I'm unable to use the GPU Kernel for dynamic partition right now, and thus it's very slow.  Long term this will be switched
            reversed_bytes = tf.reverse(the_bytes, axis=[2])

            properly_arranged_bytes = tf.where(turn_white, the_bytes, reversed_bytes)

            float_bool_masks = tf.constant(
                [np.unpackbits(num).tolist() for num in np.arange(2 ** 8, dtype=np.uint8)],
                dtype=tf.float32)

            data = tf.gather(float_bool_masks, properly_arranged_bytes)

            properly_aranged_data = tf.transpose(data, perm=[0, 2, 3, 1])

            estimator_spec = self._call_model_fn({"feature":properly_aranged_data}, None, model_fn_lib.ModeKeys.PREDICT, None)
            predictions = self._extract_keys(estimator_spec.predictions, predict_keys)[predict_keys]

            squeezed_predictions = tf.squeeze(predictions, axis=1)

            mon_sess = training.MonitoredSession(
                    session_creator=training.ChiefSessionCreator(
                        checkpoint_filename_with_path=checkpoint_path,
                        scaffold=estimator_spec.scaffold,
                        # config=self._session_config),
                        config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.5))),
                    hooks=hooks)

            def predict(the_kings, the_queens, the_rooks, the_bishops, the_knights, the_pawns, the_castling_rights, the_ep_square, the_occupied_w, the_occupied_b, the_occupied, the_turn):
                if mon_sess.should_stop():
                    raise StopIteration

                return mon_sess.run(squeezed_predictions,
                             {kings : the_kings,
                              queens : the_queens,
                              rooks : the_rooks,
                              bishops : the_bishops,
                              knights : the_knights,
                              pawns : the_pawns,
                              castling_rights : the_castling_rights,
                              ep_square : the_ep_square,
                              occupied_w : the_occupied_w,
                              occupied_b : the_occupied_b,
                              occupied : the_occupied,
                              turn_white : the_turn},
                             options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                             run_metadata=run_metadata)


            def finish():
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('timeline_eval.json', 'w') as f:
                    f.write(chrome_trace)
                mon_sess.close()

            return predict, finish




class MoveChEstimator(Estimator):

    def create_predictor(self,
                predict_keys=None,
                hooks=None,
                checkpoint_path=None):

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
        ep_square = tf.placeholder(tf.int32, shape=[None], name="ep_square_placeholder")  # NEED TO MAKE SURE THIS DOESN'T CONVERT SOME uint8 NUMBERS TO NEGATIVE int32 NUMBERS
        occupied_w = tf.placeholder(tf.int64, shape=[None], name="occupied_w_placeholder")
        occupied_b = tf.placeholder(tf.int64, shape=[None], name="occupied_b_placeholder")
        occupied = tf.placeholder(tf.int64, shape=[None], name="occupied_placeholder")
        turn_white = tf.placeholder(tf.bool, shape=[None], name="turn_placeholder")
        moves = tf.placeholder(tf.int32, shape=[None, 2], name="moves_placeholder")

        # This is overlycomplicated in regards to typing, but doesn't effect runtime speed
        bb_squares = tf.constant(np.array([np.uint64(1 << sq) for sq in range(64)] + [0 for _ in range(64, 256)],
                                          dtype=np.uint64).astype(np.int64), tf.int64)

        ep_bitboards = tf.gather(bb_squares, ep_square)

        first_players_occupied = tf.where(turn_white, occupied_b,occupied_w)  # THIS IS PURPOSELY FLIPPED!!!!!!!!!!!!!
        # first_players_occupied = tf.where(turn_white, occupied_w,occupied_b)
        second_players_occupied = tf.where(turn_white,occupied_w,occupied_b)#THIS IS PURPOSELY FLIPPED!!!!!!!!!!!!!!!!
        # second_players_occupied = tf.where(turn_white, occupied_b,occupied_w,)

        # This could likely be done using much less operations
        the_ints = tf.stack([
            tf.bitwise.invert(occupied),
            ep_bitboards,
            tf.bitwise.bitwise_and(first_players_occupied, kings),  # Can do without AND operation
            tf.bitwise.bitwise_and(first_players_occupied, queens),
            tf.bitwise.bitwise_and(first_players_occupied, rooks),
            tf.bitwise.bitwise_and(first_players_occupied, bishops),
            tf.bitwise.bitwise_and(first_players_occupied, knights),
            tf.bitwise.bitwise_and(first_players_occupied, pawns),
            tf.bitwise.bitwise_and(first_players_occupied, castling_rights),# Very likely should do this differently (to avoid 8 indices in tf.gather and instead use 1)
            tf.bitwise.bitwise_and(second_players_occupied, kings),  # Can do without AND operation
            tf.bitwise.bitwise_and(second_players_occupied, queens),
            tf.bitwise.bitwise_and(second_players_occupied, rooks),
            tf.bitwise.bitwise_and(second_players_occupied, bishops),
            tf.bitwise.bitwise_and(second_players_occupied, knights),
            tf.bitwise.bitwise_and(second_players_occupied, pawns),
            tf.bitwise.bitwise_and(second_players_occupied, castling_rights),# Very likely should do this differently (to avoid 8 indices in tf.gather and instead use 1)
        ], axis=1)

        the_bytes = tf.cast(tf.bitcast(the_ints, tf.uint8), dtype=tf.int32)

        int_turn_white = tf.cast(turn_white, tf.int32)

        to_not_reverse, to_reverse = tf.dynamic_partition(the_bytes, int_turn_white, 2)#THIS IS PURPOSELY FLIPPED!!!!!!!!!!!!!!!!!!!!!!!!!

        reversed_bytes = tf.reverse(to_reverse, axis=[2])

        bytes_shape = tf.shape(the_bytes)

        indices_to_reassemble = tf.dynamic_partition(tf.range(bytes_shape[0]), int_turn_white, 2)#THIS IS PURPOSELY FLIPPED!!!!!!!!!!!!!!!


        properly_arranged_bytes = tf.dynamic_stitch(indices_to_reassemble, [to_not_reverse,reversed_bytes])

        float_bool_masks = tf.constant(
            [np.unpackbits(num).tolist() for num in np.arange(2 ** 8, dtype=np.uint8)],
            dtype=tf.float32)

        data = tf.gather(float_bool_masks, properly_arranged_bytes)

        properly_aranged_data = tf.transpose(data, perm=[0, 2, 3, 1])


        estimator_spec = self._call_model_fn({"data":properly_aranged_data}, None, model_fn_lib.ModeKeys.PREDICT, None)

        predictions = self._extract_keys(estimator_spec.predictions, predict_keys)[predict_keys]

        legal_move_scores = tf.gather_nd(predictions, moves)


        mon_sess = training.MonitoredSession(
                session_creator=training.ChiefSessionCreator(
                    checkpoint_filename_with_path=checkpoint_path,
                    scaffold=estimator_spec.scaffold,
                    # config=self._session_config),
                    config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.5))),
                hooks=hooks)

        def predict(the_kings, the_queens, the_rooks, the_bishops, the_knights, the_pawns, the_castling_rights, the_ep_square, the_occupied_w, the_occupied_b, the_occupied, the_turn, the_moves):
            if mon_sess.should_stop():
                raise StopIteration

            return mon_sess.run(legal_move_scores,
                         {kings : the_kings,
                          queens : the_queens,
                          rooks : the_rooks,
                          bishops : the_bishops,
                          knights : the_knights,
                          pawns : the_pawns,
                          castling_rights : the_castling_rights,
                          ep_square : the_ep_square,
                          occupied_w : the_occupied_w,
                          occupied_b : the_occupied_b,
                          occupied : the_occupied,
                          turn_white : the_turn,
                          moves : the_moves},
                         options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                         run_metadata=run_metadata)


        def finish():
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()

            with open('timeline_move_scoring.json', 'w') as f:
                f.write(chrome_trace)

            mon_sess.close()

        return predict, finish