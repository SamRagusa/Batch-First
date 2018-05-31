import numpy as np
import numba as nb
import tensorflow as tf

from chess.polyglot import zobrist_hash
import random
import itertools
import chess
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import batch_first as bf

from batch_first.chestimator import new_get_board_data

from batch_first.board_jitclass import create_board_state_from_fen, traditional_perft_test

from batch_first.numba_board import create_node_info_from_fen, structured_scalar_perft_test, scalar_is_legal_move, \
    numpy_node_info_dtype, create_node_info_from_python_chess_board, push_moves

from batch_first.numba_negamax_zero_window import struct_array_to_ann_input_all_white


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


DEFAULT_MOVES_TO_CHECK = np.array(
    [[squares[0][0], squares[0][1], squares[1]] for squares in itertools.product(
        itertools.permutations(chess.SQUARES,r=2),
        [0] + [piece_type for piece_type in chess.PIECE_TYPES if not piece_type in [chess.KING, chess.PAWN]])],
    dtype=np.uint8)




def full_perft_tester(perft_function_to_test, fens_to_test=DEFAULT_TESTING_FENS, perft_results=DEFAULT_FEN_PERFT_RESULTS, max_expected_boards_to_test=5000000):
    """
    Tests a given PERFT function by checking it's results against known results for several different boards and depths.


    :param perft_function_to_test: A function that takes 2 arguments, the first being a board's FEN representation
    as a string, and the second being the depth of the perft test to be done.
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




def inference_input_pipeline_test(input_pipe_fn, boards_to_input_list_fn, fens_to_test, uses_unoccupied, piece_to_filter_fn, ep_filter_index, castling_filter_indices, flip_files=False):
    """
    A function used to test the conversion of a board to a representation consumed by ANNs during inference.
    It creates an array of features (one-hot inputs potentially with empty squares) by using the
    python-chess package, and confirms that the given pipeline produces matching inputs.


    :param input_pipe_fn: A function with no parameters, returning a size 2 tuple, with the first element being
    a tuple of Placeholders to feed values to, and the second being the tensor for the one-hot representation
    of the boards to be given to the ANN for inference
    :param boards_to_input_list_fn: A function taking as it's input a list of python-chess Board objects, and returning
    an iterable of arrays to be fed to the TensorFlow part of the pipe to test
    :param fens_to_test: The list of fen representations of game boards, as strings
    :param uses_unoccupied: A boolean value, indicating if the input to the ANNs should have a filter for
    unoccupied squares
    :param piece_to_filter_fn: A function taking a python-chess Piece as input, and returning the index of the input
     filter used to identify that piece (incremented by one if unoccupied is not used as a filter)
    :param ep_filter_index: The index of the input filter used to identify if a square is an ep square,
    this is incremented by one if unoccupied is not used as a filter
    :param castling_filter_indices:  The index of the input filter used to identify if a square contains a rook which
    has castling rights, this is incremented by one if unoccupied is not used as a filter
    :param flip_files: A boolean value indicating if the inputs to the ANNs should have it's files flipped
    :return: A size 2 tuple, the first element being a boolean value signifying if the test was passed,
    the second value is a mask of which boards passed the test
    """
    CORNER_SQUARES = np.array([bf.A1, bf.H1, bf.A8, bf.H8], np.uint8)
    CORNER_BBS = np.array([bf.BB_A1, bf.BB_H1, bf.BB_A8, bf.BB_H8], np.uint64)

    desired_filters = np.empty([len(fens_to_test), 8, 8], dtype=np.uint8)
    boards = list(map(chess.Board, fens_to_test))
    for j, fen in enumerate(fens_to_test):

        piece_map = map(boards[j].piece_at,chess.SQUARES)

        if boards[j].turn:
            cur_filter_squares = np.array(list(map(piece_to_filter_fn, piece_map)))
        else:
            cur_filter_squares = np.array(list(map(
                lambda p : piece_to_filter_fn(None if p is None else chess.Piece(p.piece_type, not p.color)),
                piece_map)))

        if not boards[j].ep_square is None:
            cur_filter_squares[boards[j].ep_square] =  ep_filter_index


        if boards[j].turn:
            CORNER_FILTER_NUMS = np.array([castling_filter_indices[j] for j in [0, 0, 1, 1]], dtype=np.uint8)
        else:
            CORNER_FILTER_NUMS = np.array([castling_filter_indices[j] for j in [1, 1, 0, 0]], dtype=np.uint8)


        castling_mask = np.array(np.uint64(boards[j].castling_rights) & CORNER_BBS,dtype=np.bool_)
        cur_filter_squares[CORNER_SQUARES[castling_mask]] =  CORNER_FILTER_NUMS[castling_mask]

        desired_filters[j] = cur_filter_squares.reshape((8,8))

        if not boards[j].turn:
            desired_filters[j] = desired_filters[j,::-1]

    if flip_files:
        desired_filters = desired_filters[...,::-1]



    #The TensorFlow code is written such that specific values can easily be pulled during debugging
    with tf.Session() as sess:
        pipe_placeholders, pipe_output = input_pipe_fn()

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

        # expected_filters, calculated_filters = sess.run([desired_square_values,filters_used],{
        #         desired_square_values: desired_filters,
        #         **dict(zip(pipe_placeholders, boards_to_input_list_fn(boards))),
        #     })
        #
        # print("Expected working method to flip a board succeeds:", np.all(expected_filters[4]==expected_filters[3]))

        problemed_board_mask = sess.run(
            boards_with_issues,
            {
                desired_square_values: desired_filters,
                **dict(zip(pipe_placeholders, boards_to_input_list_fn(boards))),
            })


        return not np.any(problemed_board_mask), problemed_board_mask






@nb.njit
def test_constants_are_different_jitted():
    """
    This is to verify that float point precision doesn't end up equating win/loss values with unreachable constants.

    :return: True if the test is passed, False if not
    """
    return  bf.MIN_FLOAT32_VAL != bf.ALMOST_MIN_FLOAT_32_VAL and bf.MAX_FLOAT32_VAL != bf.ALMOST_MAX_FLOAT_32_VAL



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
    return struct_array_to_ann_input_all_white(
        np.concatenate([
            create_node_info_from_python_chess_board(board) for board in boards]))

def cur_piece_to_filter_fn(piece):
    if piece is None:
        return 0
    if piece.color:
        return 8-piece.piece_type
    else:
        return 15-piece.piece_type


def full_test():

    result_str = ["Failed", "Passed"]

    print("Starting tests.\n")

    constants_results = test_constants_are_different_jitted()

    print("Constants test:                             %s"%result_str[constants_results])
    
    jitclass_perft_results = full_perft_tester(
        lambda fen, depth: traditional_perft_test(create_board_state_from_fen(fen), depth))

    print("PERFT test using JitClass:                  %s"%result_str[jitclass_perft_results])


    scalar_perft_results = full_perft_tester(
        lambda fen, depth: structured_scalar_perft_test(create_node_info_from_fen(fen, 0, 0), depth), max_expected_boards_to_test=20000000)

    print("PERFT test using NumPy structured scalar:   %s"%result_str[scalar_perft_results])


    move_verification_results = move_verification_tester(
        scalar_is_legal_move,
        lambda b: create_node_info_from_python_chess_board(b)[0])

    print("Move legality verification test:            %s"%result_str[move_verification_results])

    zobrist_hash_results = zobrist_hash_test(cur_hash_getter,num_sequences_to_test=5000, max_moves_per_test=40)

    print("Incremental Zobrist hash test:              %s"%result_str[zobrist_hash_results])

    input_pipe_results = inference_input_pipeline_test(
        new_get_board_data,
        cur_boards_to_input_list_fn,
        DEFAULT_TESTING_FENS,
        False,
        cur_piece_to_filter_fn,
        ep_filter_index=1,
        castling_filter_indices=[8, 15])

    if not input_pipe_results[0]:
        flipped_input_pipe_results = inference_input_pipeline_test(
            new_get_board_data,
            cur_boards_to_input_list_fn,
            DEFAULT_TESTING_FENS,
            False,
            cur_piece_to_filter_fn,
            ep_filter_index=1,
            castling_filter_indices=[8, 15],
            flip_files=True)

        print("Input pipeline test:                        %s but..."%result_str[input_pipe_results[0]])
        print("Input pipeline with flipped files test:     %s"%result_str[flipped_input_pipe_results[0]])
        input_pipe_results = flipped_input_pipe_results
    else:
        print("Input pipeline test:                        %s" % result_str[input_pipe_results[0]])



    if constants_results and jitclass_perft_results and scalar_perft_results and move_verification_results and zobrist_hash_results and input_pipe_results[0]:
        print("\nAll tests were passed!")
        return True
    else:
        print("\nSome tests failed! (see above).")
        return False












full_test()