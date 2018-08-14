import tensorflow as tf

import random
import chess.pgn
import os
import tempfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from batch_first import *

from batch_first.anns.database_creator import create_database_from_pgn, known_scoring_writer_creator

from batch_first.chestimator import get_board_data

from batch_first.board_jitclass import create_board_state_from_fen, traditional_perft_test

from batch_first.numba_board import create_node_info_from_fen, structured_scalar_perft_test, scalar_is_legal_move, \
    numpy_node_info_dtype, create_node_info_from_python_chess_board, push_moves, set_up_move_array, flip_vertically

from batch_first.numba_negamax_zero_window import struct_array_to_ann_inputs, has_insufficient_material, \
    set_up_root_node_for_struct, zero_window_negamax_search

from batch_first.transposition_table import get_empty_hash_table

from batch_first.global_open_priority_nodes import PriorityBins



#These fens come from the ChessProgramming Wiki, and can be found here: https://chessprogramming.wikispaces.com/Perft+Results
DEFAULT_TESTING_FENS = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
                        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", #In check
                        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", #Mirrored version of above
                        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8 ",
                        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"]


#These values come from the ChessProgramming Wiki, and can be found here: https://chessprogramming.wikispaces.com/Perft+Results
DEFAULT_FEN_PERFT_RESULTS = [[20,400,8902,197281,4865609,119060324,3195901860,849989789566,2439530234167,693552859712417],
                             [48,2039,97862,4085603,193690690],
                             [14,191,2812,43238,674624,11030083, 178633661],
                             [6,264,9467,422333,15833292,706045033],
                             [6,264,9467,422333,15833292,706045033],
                             [44,1486,62379,2103487,89941194],
                             [46,2079,89890,3894594,164075551,6923051137,287188994746,11923589843526,490154852788714]]


# This array represents moves, some illegal or impossible, for use in testing.  It contains all possible combinations
# of from squares, to squares, and promotion pieces.  The array is of the form:
# [..., [move_from_square, move_to_square, move_promotion], ...]
DEFAULT_MOVES_TO_CHECK = np.array(
    [[squares[0][0], squares[0][1], squares[1]] for squares in itertools.product(
        itertools.permutations(chess.SQUARES,r=2),
        [0] + [piece_type for piece_type in chess.PIECE_TYPES if not piece_type in [chess.KING, chess.PAWN]])],
    dtype=np.uint8)




def full_perft_tester(perft_function_to_test, fens_to_test=DEFAULT_TESTING_FENS, perft_results=DEFAULT_FEN_PERFT_RESULTS, max_expected_boards_to_test=5000000):
    """
    Tests a given PERFT function by checking it's results against known results for several different boards and depths.


    :param perft_function_to_test: A function that takes 2 arguments, the first being a board's FEN representation
     as a string, and the second being the depth of the PERFT test to be done.
    :param fens_to_test: An iterable of strings, each a FEN representation of a board.
    :param perft_results: A list of the same length as fens_to_test, containing lists of integers, each corresponding
     to the expected PERFT result if tested at a depth of it's index
    :param max_expected_boards_to_test: The maximum expected PERFT result to test for a given board
    :return: True if all tests were passed, False if not
    """
    for cur_fen, cur_results in zip(fens_to_test, perft_results):
        for j, expected_result in enumerate(cur_results):
            if expected_result <= max_expected_boards_to_test:
                if perft_function_to_test(cur_fen, j+1) != expected_result:
                    return False
    return True


def zobrist_hash_test(hash_getter, fen_to_start=None, num_sequences_to_test=1000, max_moves_per_test=20):
    """
    This functions tests the engine's ability to incrementally maintain a board's Zobrist hash while pushing
    moves.  It tests this by generating a set of random move sequences from a given board, and compares the computed
    results with the values computed by the functions within the python-chess package.


    :param hash_getter: The hashing function to test, it must accept 3 parameters, the fen to start the
     move sequences from, the list of lists of python-chess Moves to make from the starting board, and the maximum
     number of moves per test. The function must return an ndarray of the board hashes at each position in the random
     move sequences (zero if the sequence terminates early), it's shape will be
     [num_sequences_to_test, max_moves_per_test]
    :param fen_to_start: The fen (as a string) representing the board to start making random moves from.  If None is
     given, the fen for the start of a normal game is used
    :param num_sequences_to_test: The number of random move sequences (each starting from the given initial fen) to test
    :param max_moves_per_test: The maximum number of random moves to be made for each testing sequence.  This is a
     maximum because some random move sequences result in a premature win/loss/draw
    :return: True if all tests were passed, False if not
    """
    if fen_to_start is None:
        fen_to_start = DEFAULT_TESTING_FENS[0]

    correct_hashes = np.zeros((num_sequences_to_test, max_moves_per_test), dtype=np.uint64)
    move_lists = [[] for _ in range(num_sequences_to_test)]
    for j in range(num_sequences_to_test):
        cur_board = chess.Board(fen_to_start)
        for i in range(max_moves_per_test):
            possible_next_moves = list(cur_board.generate_legal_moves())
            if len(possible_next_moves) == 0:
                break
            move_lists[j].append(possible_next_moves[random.randrange(len(possible_next_moves))])
            cur_board.push(move_lists[j][-1])

            correct_hashes[j,i] = zobrist_hash(cur_board)

    #Go through incorrect hashes and print relevant information about them for use during debugging
    calculated_hashes = hash_getter(fen_to_start, move_lists, max_moves_per_test)
    same_hashes = calculated_hashes == correct_hashes
    if not np.all(same_hashes):
        for j in range(len(same_hashes)):
            if np.sum(same_hashes[j]) != 0:
                cur_board = chess.Board(fen_to_start)
                for i,move in enumerate(move_lists[j]):
                    if not same_hashes[j,i]:
                        print("Board and move being made which caused the first incorrect hash in sequence %d:\n%s\n%s\n%s\nDifference in hash values:%d\n" % (
                            j, cur_board, cur_board.fen(), move, correct_hashes[j, i] ^ calculated_hashes[j, i]))
                        break

                    cur_board.push(move)

    return np.all(same_hashes)


def get_expected_features(boards, piece_to_filter_fn, ep_filter_index, castling_filter_indices):
    """
    Gets an array of feature indices for the given chess boards.


    :param boards: A list of of python-chess Board objects.  These are the boards the expected features
     will be computed for
    :param piece_to_filter_fn: A function taking a python-chess Piece as input, and returning the index of the input
     filter used to identify that piece (incremented by one if unoccupied is not used as a filter)
    :param ep_filter_index: The index of the input filter used to identify if a square is an ep square,
     this is incremented by one if unoccupied is not used as a filter
    :param castling_filter_indices: The index of the input filter used to identify if a square contains a rook which
     has castling rights, this is incremented by one if unoccupied is not used as a filter
    :return: An ndarray containing the given boards features, it will have size [len(boards),8,8] and dtype np.uint8
    """
    CORNER_SQUARES = np.array([A1, H1, A8, H8], np.uint8)
    CORNER_BBS = np.array([BB_A1, BB_H1, BB_A8, BB_H8], np.uint64)

    desired_filters = np.empty([len(boards), 8, 8], dtype=np.uint8)


    for j in range(len(boards)):

        piece_map = map(boards[j].piece_at,chess.SQUARES)

        if boards[j].turn:
            cur_filter_squares = np.array(list(map(piece_to_filter_fn, piece_map)))
        else:
            cur_filter_squares = np.array(list(map(
                lambda p : piece_to_filter_fn(None if p is None else chess.Piece(p.piece_type, not p.color)),
                piece_map)))

        if not boards[j].ep_square is None:
            cur_filter_squares[boards[j].ep_square] = ep_filter_index

        if boards[j].turn:
            CORNER_FILTER_NUMS = np.array([castling_filter_indices[j] for j in [0, 0, 1, 1]], dtype=np.uint8)
        else:
            CORNER_FILTER_NUMS = np.array([castling_filter_indices[j] for j in [1, 1, 0, 0]], dtype=np.uint8)


        castling_mask = np.array(np.uint64(boards[j].castling_rights) & CORNER_BBS,dtype=np.bool_)
        cur_filter_squares[CORNER_SQUARES[castling_mask]] =  CORNER_FILTER_NUMS[castling_mask]

        desired_filters[j] = cur_filter_squares.reshape((8,8))

        if not boards[j].turn:
            desired_filters[j] = desired_filters[j,::-1]

    return desired_filters


def inference_input_pipeline_test(inference_pipe_fn, boards_to_input_list_fn, boards, uses_unoccupied, piece_to_filter_fn, ep_filter_index, castling_filter_indices):
    """
    A function used to test the conversion of a board to a representation consumed by ANNs during inference.
    It creates an array of features (one-hot inputs potentially with empty squares) by using the
    python-chess package, and confirms that the given pipeline produces matching inputs.


    :param inference_pipe_fn: A function with no parameters, returning a size 2 tuple, with the first element being
     a tuple of Placeholders to feed values to, and the second being the tensor for the one-hot representation
     of the boards to be given to the ANN for inference
    :param boards_to_input_list_fn: A function taking as it's input a list of python-chess Board objects, and returning
     an iterable of arrays to be fed to the TensorFlow part of the pipe to test
    :param boards: A list of of python-chess Board objects
    :param uses_unoccupied: A boolean value, indicating if the input to the ANNs should have a filter for
     unoccupied squares
    :param piece_to_filter_fn: A function taking a python-chess Piece as input, and returning the index of the input
     filter used to identify that piece (incremented by one if unoccupied is not used as a filter)
    :param ep_filter_index: The index of the input filter used to identify if a square is an ep square,
     this is incremented by one if unoccupied is not used as a filter
    :param castling_filter_indices:  The index of the input filter used to identify if a square contains a rook which
     has castling rights, this is incremented by one if unoccupied is not used as a filter
    :return: A size 3 tuple, the first element being a boolean value signifying if the test was passed,
     the second element being the expected filters, and the third being the filters computed by the inference pipeline
    """

    desired_filters = get_expected_features(boards, piece_to_filter_fn, ep_filter_index, castling_filter_indices)

    #The TensorFlow code is written such that specific values can easily be pulled during debugging
    with tf.Session() as sess:
        pipe_placeholders, pipe_output = inference_pipe_fn()

        desired_square_values = tf.placeholder(tf.uint8, [None,8,8])

        # Using absolute value so that an array such as [1,0,-1,1] isn't perceived as a one-hot vector
        abs_filter_sums = tf.reduce_sum(tf.abs(pipe_output), axis=3)

        squares_with_incorrect_sums = tf.logical_and(tf.not_equal(abs_filter_sums,0), tf.not_equal(abs_filter_sums, 1))

        board_has_square_with_incorrect_filter_sum = tf.reduce_any(squares_with_incorrect_sums, axis=[1, 2])

        if uses_unoccupied:
            onehot_data = pipe_output
        else:
            onehot_data = tf.concat([tf.expand_dims(1-abs_filter_sums,axis=3), pipe_output], axis=3)


        filters_used = tf.cast(tf.argmax(onehot_data,axis=3),tf.uint8)

        wrong_filter_squares = tf.not_equal(desired_square_values, filters_used)

        board_has_wrong_filter = tf.reduce_any(wrong_filter_squares, axis=[1,2])

        boards_with_issues =  tf.logical_or(board_has_square_with_incorrect_filter_sum, board_has_wrong_filter)


        calculated_filters, problemed_board_mask = sess.run(
            [filters_used, boards_with_issues],
            {
                desired_square_values: desired_filters,
                **dict(zip(pipe_placeholders, boards_to_input_list_fn(boards))),
            })

        return not np.any(problemed_board_mask), desired_filters, calculated_filters


def move_verification_tester(move_legality_tester, board_creator_fn, fens_to_test=DEFAULT_TESTING_FENS, moves_to_test=DEFAULT_MOVES_TO_CHECK):
    """
    A function to test a method of move legality verification.  This is used to confirm that the move verification done
    for moves stored in the transposition table is correct.  It uses the python-chess package to compute the correct
    legality of moves, and compares against that.


    :param move_legality_tester: A function accepting two parameters, the first being a board as returned by the given
     board_creator_fn, and the second being a size 3 ndarray representing a move. The function must return a
     boolean value, indicating if the move is legal
    :param board_creator_fn: A function which takes as input a Python-Chess Board object, and outputs the board in the
     representation to be used when testing move legality.
    :param fens_to_test: An iterable of strings, each a FEN representation of a board.
    :param moves_to_test: A uint8 ndarray of size [num_moves_to_test, 3], representing the moves to test for each
     testing fen
    :return: True if all tests were passed, False if not
    """
    for cur_fen in fens_to_test:
        cur_board = chess.Board(cur_fen)

        cur_testing_board = board_creator_fn(cur_board)
        for j in range(len(moves_to_test)):
            if move_legality_tester(cur_testing_board, moves_to_test[j]) != cur_board.is_legal(
                    chess.Move(*moves_to_test[j]) if moves_to_test[j, 2] != 0 else chess.Move(*moves_to_test[j, :2])):
                return False

    return True


def complete_board_eval_tester(tfrecords_writer, feature_getter, inference_pipe_fn, boards_to_input_for_inference,
                                   piece_to_filter_fn, ep_filter_index, castling_filter_indices, num_random_games=20,
                                   max_moves_per_game=250, uses_unoccupied=False):

    """
    This function can be thought of in two parts, one tests the board evaluation inference pipeline, and the other
     tests the entire board evaluation training pipeline (from PGN to Tensor).

    It works by first playing a number of chess games, choosing moves randomly and storing each board configuration.
     It then uses inference_input_pipeline_test to both test the inference pipeline, and produce the expected
     feature arrays for the training pipeline.  A temporary pgn file is created from the games played,
     and sent through the pipeline used to create and consume the board evaluation database.  If every feature array
     produced by the training pipeline is contained within the expected feature arrays,
     then the training pipeline passes the test.



    :param tfrecords_writer: A function to write a tfrecords file in the same way done when creating the board
     evaluation ANN training data.  It must accept two parameters, the first is the filename of the pgn file,
     and the second the desired output tfrecords filename.  The function doesn't need to return anything
    :param feature_getter: A function accepting the filename of the tfrecords file created for testing,
     and returning the tensor for the one-hot representation of the boards to be given to the ANN
     for training (unoccupied filter may be omitted if not used)
    :param inference_pipe_fn: See inference_input_pipeline_test
    :param boards_to_input_for_inference: See inference_input_pipeline_test
    :param piece_to_filter_fn: See inference_input_pipeline_test
    :param ep_filter_index: See inference_input_pipeline_test
    :param castling_filter_indices: See inference_input_pipeline_test
    :param num_random_games: The number of chess games to be played/tested (choosing random moves).  This is used for
     creating the PGN file to test the training pipe, or in generating the list of boards to test the inference pipe
    :param max_moves_per_game: The maximum number of moves (halfmoves) to be made before a game should be stopped.
     If None is given, a limit of 250 will be used
    :param uses_unoccupied: See inference_input_pipeline_test
    :return: A tuple of two boolean values, the first indicating if the training pipeline tests were passed,
     and the second indicating if the inference pipeline tests were passed


    NOTES:
    1) Ideally a temporary file would be used for writing/reading the tfrecords database
    """
    tf_records_filename = "THIS_COULD-SHOULD_BE_A_TEMPORARY_FILE.tfrecords"

    temp_pgn, temp_filename = tempfile.mkstemp()

    boards = [chess.Board()]
    for j in range(num_random_games):

        cur_board = chess.Board()
        for _ in range(max_moves_per_game):
            possible_next_moves = list(cur_board.generate_legal_moves())

            cur_board.push(possible_next_moves[random.randrange(len(possible_next_moves))])

            boards.append(cur_board.copy())

            if cur_board.is_game_over():
                break

        #Writing the game to the pgn file
        os.write(temp_pgn, str.encode(str(chess.pgn.Game.from_board(cur_board))))

    #Create the tfrecords file as it would be created for training
    tfrecords_writer(temp_filename, tf_records_filename)

    os.close(temp_pgn)


    input_features = feature_getter(tf_records_filename)

    one_hot_features = np.argmax(np.concatenate((np.expand_dims(1-np.sum(input_features, axis=3),axis=3),input_features), axis=3),axis=3)


    inference_pipe_results, desired_filters, inference_filters = inference_input_pipeline_test(
        inference_pipe_fn,
        boards_to_input_for_inference,
        boards,
        uses_unoccupied,
        piece_to_filter_fn,
        ep_filter_index=ep_filter_index,
        castling_filter_indices=castling_filter_indices)

    no_training_pipe_issues = True
    for filter in one_hot_features:
        num_correct_squares = np.sum(filter == desired_filters, (1,2))
        if np.all(64 != num_correct_squares):


            print(filter)

            max_correct = np.max(num_correct_squares)
            print(desired_filters[num_correct_squares==max_correct])
            in_boards = np.array(boards)[num_correct_squares == max_correct][0]
            print(in_boards.fen())
            print(in_boards, "\n")

            no_training_pipe_issues = False

    return no_training_pipe_issues, inference_pipe_results


def zero_window_search_tester(expected_val_fn, calculated_evaluator, hash_table_creator, boards=None, fens=None, max_depth=3, max_separator_change=.1, runs_per_depth=5):
    """
    A function to test a zero-window minimax search.  It first computes a 'correct' minimax value based on
    a given search function, then repeatedly does zero-window search calls (with increasing search depth)
    above and below the 'correct' minimax value to verify that the searches are behaving as desired.



    :param expected_val_fn: A function which calculates and returns the full minimax value of a board.  It must accept
    2 parameters, a Python-Chess board, and the desired search depth.
    :param calculated_evaluator: The zero-window search function being tested.  It requires 4 parameters,
    a Python-Chess board, the depth to search, the separator to test, and a hash table.  It must return
    the value calculated
    :param hash_table_creator: A function which takes no arguments and returns a hash table to give the search being
    tested
    :param boards: An iterable of Python-Chess Board objects.   If supplied, 'fens' parameter will be ignored
    :param fens: An iterable of strings, each a FEN representation of a board.  If 'boards' is supplied this will not be
    used, and if neither are supplied DEFAULT_TESTING_FENS will be used
    :param max_depth: The maximum search depth to test.  Tests will be performed for depths 1 to max_depth
    :param max_separator_change: The maximum distance from the expected minimax value to do a zero-window test
    :param runs_per_depth: The number of times the zero-window search will be tested for each board/depth combination
    (one 'run' is testing one value above the minimax value, and one value below)
    :return: True if all tests were passed, False if not
    """
    issue_printer = lambda b,m,s,c : print(
        "%s\n%s\nThe actual minimax value for the above board is %f, but when testing a value of %f, the zero-window search resulted in %f"%(b, b.fen(), m, s, c))

    def test_board(board):
        hash_table = hash_table_creator()
        for depth in range(1, max_depth+1):
            minimax_value = expected_val_fn(board, depth)

            for _ in range(runs_per_depth):
                above_val_separator = minimax_value + np.random.rand() * max_separator_change

                if above_val_separator == minimax_value:
                    above_val_separator = np.nextafter(np.nextafter(minimax_value, MAX_FLOAT32_VAL), MAX_FLOAT32_VAL)

                calculated_value = calculated_evaluator(board, depth, above_val_separator, hash_table)

                if calculated_value < minimax_value or calculated_value >= above_val_separator:
                    issue_printer(board, minimax_value, above_val_separator, calculated_value)
                    return False

                below_val_separator = minimax_value - np.random.rand() * max_separator_change

                if below_val_separator == minimax_value:
                    below_val_separator = np.nextafter(np.nextafter(minimax_value, MIN_FLOAT32_VAL), MIN_FLOAT32_VAL)

                calculated_value = calculated_evaluator(board, depth, below_val_separator, hash_table)

                if calculated_value > minimax_value or calculated_value < below_val_separator:
                    issue_printer(board, minimax_value, below_val_separator, calculated_value)
                    return False
        return True


    if boards is None:
        boards = map(chess.Board, fens if fens else DEFAULT_TESTING_FENS)

    return all(map(test_board, boards))









def dummy_eval_for_simple_search(board):
    dummy_metric = np.bitwise_or(board['occupied_w'], board['occupied_b'])
    if not board['turn']:
        dummy_metric = flip_vertically(dummy_metric)
    return dummy_metric.astype(np.float32) / BB_ALL.astype(np.float32)


def dummy_eval_for_bf(*args):
    return np.bitwise_or(args[1][:,0], args[1][:,1]).astype(np.float32)/BB_ALL.astype(np.float32)


def create_negamax_function(eval_fn):
    """
    :param eval_fn: The evaluation function used must account for how the board data (in batch first) is given as if
    it were white's turn (for the sake of the ANNS), and must match that behavior
    (see 'dummy_eval_for_bf' and 'dummy_eval_for_simple_search')
    """
    def get_eval_score(board, depth):
        if board["halfmove_clock"] >= 50 or has_insufficient_material(board):
            return TIE_RESULT_SCORE

        set_up_move_array(board)

        if board['children_left'] == 0:
            if board['best_value'] == TIE_RESULT_SCORE:
                return TIE_RESULT_SCORE
            return LOSS_RESULT_SCORES[depth] if board['turn'] else WIN_RESULT_SCORES[depth]

        if depth == 0:
            return eval_fn(board)

        return None

    def simple_struct_copy_push(board, move):
        to_push = np.array([board.copy()])
        push_moves(to_push, np.array([move]))

        to_return = to_push[0]
        to_return['children_left'] = 0
        to_return['unexplored_moves'][:] = 255

        return to_return

    def negamax(board, depth, alpha, beta, color):
        eval_score = get_eval_score(board, depth)

        if not eval_score is None:
            return color * eval_score

        best_val = MIN_FLOAT32_VAL
        for j in range(board['children_left']):
            best_val = np.maximum(
                best_val,
                - negamax(simple_struct_copy_push(board, board['unexplored_moves'][j]), depth - 1, -beta, -alpha, -color))

            alpha = np.maximum(alpha, best_val)
            if alpha >= beta:
                break
        return best_val

    def search_helper(py_board, depth, alpha=MIN_FLOAT32_VAL, beta=MAX_FLOAT32_VAL):
        color = 1 if py_board.turn else -1
        return negamax(create_node_info_from_python_chess_board(py_board)[0], depth, alpha, beta, color)

    return search_helper


def negamax_zero_window_search_creator(move_predictor, max_batch_size=5000, run_search_in_testing_mode=False):
    def zero_window_search(board, depth, separator, hash_table):
        priority_bins = PriorityBins(
            np.linspace(0,1,1000),
            max_batch_size,
            testing=run_search_in_testing_mode)

        separator_to_use = np.nextafter(separator, MIN_FLOAT32_VAL)

        root_node = set_up_root_node_for_struct(
            move_predictor,
            hash_table,
            create_node_info_from_python_chess_board(board, depth, separator_to_use))

        if root_node.board_struct[0]['terminated']:
            return root_node.board_struct[0]['best_value']

        if bool(depth % 2) == board.turn:
            board_eval_fn = lambda *args: -dummy_eval_for_bf(*args)
        else:
            board_eval_fn = dummy_eval_for_bf

        to_return = zero_window_negamax_search(
            root_node,
            priority_bins,
            board_eval_fn,
            move_predictor,
            hash_table=hash_table,
            testing=run_search_in_testing_mode)
        return to_return

    return zero_window_search


def pseudo_random_move_eval(*args):
    """
    NOTES:
    1) This function uses np.linspace instead of randomly generated values to maintain the deterministic behavior of the
    tree search (only helpful when tracking down a bug).
    """
    return lambda x: np.linspace(0, 1, np.sum(x[1]))


def cur_hash_getter(fen_to_start,move_lists,max_possible_moves):
    hashes = np.zeros((len(move_lists), max_possible_moves), dtype=np.uint64)
    initial_board = create_node_info_from_fen(fen_to_start, 255, 0)
    for j in range(len(move_lists)):
        board = initial_board.copy()
        for i,move in enumerate(move_lists[j]):
            push_moves(board, np.array([[move.from_square, move.to_square, 0 if move.promotion is None else move.promotion]], dtype=np.uint8))
            hashes[j,i] = board[0]['hash']
    return hashes


def cur_boards_to_input_list_fn(boards):
    struct_array = np.concatenate([create_node_info_from_python_chess_board(board) for board in boards])

    return struct_array_to_ann_inputs(
        struct_array,
        np.array([], dtype=numpy_node_info_dtype),
        np.ones(len(struct_array), dtype=np.bool_),
        np.array([], dtype=np.bool_),
        len(struct_array))


def cur_piece_to_filter_fn(piece):
    if piece is None:
        return 0
    if piece.color:
        return 8-piece.piece_type
    else:
        return 15-piece.piece_type


def dummy_records_writer_creator():
    class DummyResult:
        def __init__(self):
            self.cp = 0
            self.mate = None

    def cur_tfrecords_writer(pgn_filename, output_filename):
        create_database_from_pgn(
            [pgn_filename],
            data_writer=known_scoring_writer_creator(
                score_board=lambda b : DummyResult(),
                print_info=False),
            output_filenames=[output_filename],
            print_info=False)
    return cur_tfrecords_writer


def dummy_tf_records_input_data_pipe_creator(filename, include_unoccupied=False):
    def tf_records_input_data_fn():
        dataset = tf.data.TFRecordDataset(filename)

        def parser(record):
            keys_to_features = {"board": tf.FixedLenFeature([8 * 8 ], tf.int64),
                                "score": tf.FixedLenFeature([], tf.int64)}

            parsed_example = tf.parse_single_example(record, keys_to_features)

            reshaped_board = tf.reshape(parsed_example['board'], [8,8])

            omit_unoccupied_decrement = 0 if include_unoccupied else 1
            one_hot_board = tf.one_hot(reshaped_board - omit_unoccupied_decrement, 16 - omit_unoccupied_decrement)
            return one_hot_board

        dataset = dataset.map(parser)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        return {"board" : features}

    return tf_records_input_data_fn


def cur_tfrecords_to_features(filename):
    input_generator = dummy_tf_records_input_data_pipe_creator(filename)()
    with tf.Session() as sess:
        inputs = []
        try:
            while True:
                inputs.append(sess.run(input_generator['board']))
        except tf.errors.OutOfRangeError:
            return np.stack(inputs)


def full_test():
    """
    Runs every test relevant to Batch First's performance or correctness (that's been created so far).

    :return: A boolean value indicating if all tests were passed
    """
    result_str = ["Failed", "Passed"]

    print("Starting tests (this will likely take 1-5 minutes).\n")

    test_results = [False] * 7

    test_results[0] = full_perft_tester(
        lambda fen, depth: traditional_perft_test(create_board_state_from_fen(fen), depth))

    print("PERFT test using JitClass:                                    %s" % result_str[test_results[0]])

    test_results[1] = full_perft_tester(
        lambda fen, depth: structured_scalar_perft_test(create_node_info_from_fen(fen, 0, 0), depth))

    print("PERFT test using NumPy structured scalar:                     %s" % result_str[test_results[1]])

    test_results[2] = move_verification_tester(
        scalar_is_legal_move,
        lambda b: create_node_info_from_python_chess_board(b)[0])

    print("Move legality verification test:                              %s" % result_str[test_results[2]])

    test_results[3] = zobrist_hash_test(cur_hash_getter)

    print("Incremental Zobrist hash test:                                %s" % result_str[test_results[3]])

    test_results[4], test_results[5] = complete_board_eval_tester(
        tfrecords_writer=dummy_records_writer_creator(),
        feature_getter=cur_tfrecords_to_features,
        inference_pipe_fn=get_board_data,
        boards_to_input_for_inference=cur_boards_to_input_list_fn,
        piece_to_filter_fn=cur_piece_to_filter_fn,
        ep_filter_index=1,
        castling_filter_indices=[8, 15])

    print("Board evaluation data creation and training pipeline test:    %s" % result_str[test_results[4]])
    print("Board evaluation inference pipeline test:                     %s" % result_str[test_results[5]])

    test_results[6] = zero_window_search_tester(
        create_negamax_function(dummy_eval_for_simple_search),
        negamax_zero_window_search_creator(pseudo_random_move_eval),
        get_empty_hash_table)

    print("Zero-window search test:                                      %s" % result_str[test_results[6]])


    if all(test_results):
        print("\nAll tests were passed!")
        return True
    else:
        print("\nSome tests failed! (see above).")
        return False






if __name__ == "__main__":
    full_test()