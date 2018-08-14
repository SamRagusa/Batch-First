import tensorflow as tf

import chess.pgn
import chess.uci
import pickle
import math
import time
import random

from concurrent.futures import ThreadPoolExecutor, as_completed

from batch_first.numba_board import flip_vertically, msb,  popcount, numpy_node_info_dtype, set_up_move_array, push_moves
from batch_first.board_jitclass import BoardState, Move, create_board_state_from_fen, generate_legal_moves, \
    copy_push, push_with_hash_update

# from batch_first.numba_negamax_zero_window import struct_array_to_ann_inputs
# from batch_first.chestimator import get_inference_functions

from batch_first import *



def dict_key_to_py_board(key):
    py_board = chess.Board()

    py_board.kings = int(key[0])
    py_board.queens = int(key[1])
    py_board.rooks =  int(key[2])
    py_board.bishops = int(key[3])
    py_board.knights = int(key[4])
    py_board.pawns = int(key[5])

    py_board.occupied = int(key[10])
    py_board.occupied_co[True] = int(key[8])
    py_board.occupied_co[False] = int(key[9])

    py_board.castling_rights = int(key[6])

    py_board.ep_square = int(msb(key[7])) if key[7] != 0 else None

    py_board.turn = True

    return py_board


@nb.njit
def board_state_from_dict_key(the_key):
    return BoardState(the_key[5], the_key[4], the_key[3], the_key[2], the_key[1], the_key[0], the_key[8],
                      the_key[9], the_key[10], TURN_WHITE, the_key[6], None if the_key[7]==0 else msb(the_key[7]), 0, 0)


# @nb.njit
def board_info_array_from_dict_keys(keys):
    return np.array([(key[5], key[4], key[3], key[2], key[1], key[0], key[8], key[9], key[10], TURN_WHITE,
                      key[6], 0 if key[7]==0 else msb(key[7]), 0, 0,False,0,255,
                      MIN_FLOAT32_VAL, np.full([MAX_MOVES_LOOKED_AT, 3], 255, dtype=np.uint8),
                      np.full([MAX_MOVES_LOOKED_AT], MIN_FLOAT32_VAL, dtype=np.float32),
                      np.full([3], 255, dtype=np.uint8), 0, 0) for key in keys],dtype=numpy_node_info_dtype)


@nb.njit
def get_board_info_tuple(board):
    if board.turn == TURN_WHITE:
        return np.array([board.kings, board.queens, board.rooks, board.bishops, board.knights, board.pawns,
                        board.castling_rights, 0 if board.ep_square is None else BB_SQUARES[np.int32(board.ep_square)],
                        board.occupied_w, board.occupied_b, board.occupied],dtype=np.uint64)
    else:
        return np.array([board.kings, board.queens, board.rooks, board.bishops, board.knights, board.pawns,
                        board.castling_rights, 0 if board.ep_square is None else BB_SQUARES[np.int32(board.ep_square)],
                        board.occupied_b, board.occupied_w, board.occupied],dtype=np.uint64)


def get_feature_array(white_info):
    """
    NOTES:
    1) For the method used here to work (iterating over the masks), rook indices must always be set before
    castling_rights, and unoccupied must be set before the ep_square.
    """
    answer = np.zeros(64,dtype=np.uint8)

    occupied_colors = np.array([[white_info[8]], [white_info[9]]])
    piece_info = np.array([white_info[:7]])

    masks = np.unpackbits(np.bitwise_and(occupied_colors, piece_info).reshape(14).view(np.uint8)).reshape(
        [14, 8, 8]).view(np.bool_)[..., ::-1].reshape(14, 64)


    for j, mask in enumerate(masks):
        answer[mask] = j + 2

    #Set the ep square
    if white_info[7] !=0:
        answer[msb(white_info[7])] = 1


    return answer




class BoardData(object):
    """
    A class to store information relevant to a single game board, such as moves chosen by
    players (and their occurrences).
    """

    def __init__(self, move_tuple):
        """
        Creates a BoardData object, and updates it's data with the occurrence of the given move.
        """
        self.moves = {}
        self.update(move_tuple)

    def update(self, move_tuple):
        """
        A method to update the data from a given move.  It increments the counter for that move,
        or if one does not currently exist, it creates it and sets it to one.
        """
        if self.moves.get(move_tuple) is None:
            self.moves[move_tuple] = 1
        else:
            self.moves[move_tuple] += 1


def create_database_from_pgn(filenames, game_filters=[], to_collect_filters=[], post_collection_filters=[], data_writer=None,
                             num_first_moves_to_skip=0, output_filenames=["board_config_database.csv"], print_info=True):
    """
    A function used to generate customized chess databases from a set of pgn files.  It does so using the python-chess
    package for pgn file parsing, and error handling (if the error lies within the pgn file itself), and uses
    Batch First's BoardState JitClass for move generations.


    :param filenames: An array of filenames (paths) for the png files of chess games to read data from.
    :param game_filters: A list of functions used to filter out games.  Each function mast accept a python-chess Game
     game object and return a boolean value indicating weather the game should be filtered out or not
    :param to_collect_filters: A list of functions used to filter out (board, move) pairs during the parsing of the pgn files. Each function in
     the array must accept two parameters, the first being the current BoardState object, and the second being the next
     move to be made, as a Python-Chess Move object.  They should return True if the (board, move) pair should be
     filtered out, False if not.
    :param post_collection_filters: An array of functions just like to_collect_filters, except that the second
     parameter is a BoardData object instead of a move, and the filters are applied after all pgn files have been parsed,
     as opposed to during.
    :param data_writer: A function that takes in two parameters, the first of which is a dictionary mapping string
     representations of boards (as determined by other parameters of this function) to BoardData objects,
     and the second parameter accepts the output_filenames array, which is given as another parameter of this function.
     If none is give, it will pickle and save the dictionary of information.
    :param num_first_moves_to_skip: The number of halfmoves to omit at the start of every game during collection
     of board data
    :param output_filenames: An array of filenames (paths) to be passed to the data_writer as a parameter.
    :print_info: A boolean value, indicating if updates on what is happening within the function should be printed


    NOTES:
    1) Parsing errors are handled internally within the python-chess package, and the logs are being stored
    in the Game object's error array.
    """
    def game_is_okay(game):
        for filter in game_filters:
            if filter(game):
                return False
        return True


    configs = {}
    for index, filename in enumerate(filenames):
        if print_info:
            print("Starting file", str(index + 1))

        pgn_file = open(filename)

        cur_game = chess.pgn.read_game(pgn_file)
        while not cur_game is None and not game_is_okay(cur_game):
            cur_game = chess.pgn.read_game(pgn_file)

        while not cur_game is None:
            the_board = create_board_state_from_fen(INITIAL_BOARD_FEN)

            for move_num, move in enumerate(cur_game.main_line()):
                should_save = move_num >= num_first_moves_to_skip
                if should_save and to_collect_filters != []:
                    for filter in to_collect_filters:
                        if filter(the_board, move):
                            should_save = False
                            break

                if should_save:
                    if the_board.turn == TURN_WHITE:
                        white_move_info = tuple(get_board_info_tuple(the_board))
                    else:
                        white_move_info = tuple(flip_vertically(get_board_info_tuple(the_board)))

                    if configs.get(white_move_info) is None:
                        configs[white_move_info] = BoardData((move.from_square, move.to_square, move.promotion))
                    else:
                        configs[white_move_info].update((move.from_square, move.to_square, move.promotion))

                push_with_hash_update(the_board, Move(move.from_square, move.to_square, 0 if move.promotion is None else move.promotion))


            cur_game = chess.pgn.read_game(pgn_file)
            while not cur_game is None and not game_is_okay(cur_game):
                cur_game = chess.pgn.read_game(pgn_file)

        pgn_file.close()

    if print_info:
        print("Applying post-collection filters to data.")

    to_delete = []
    if post_collection_filters != []:
        for board_info, data in configs.items():
            for filter in post_collection_filters:
                if filter(board_info, data):
                    to_delete.append(board_info)
                    break

        if print_info:
            print("Number of boards deleted by post-collection filters:", len(to_delete))

        for board_info in to_delete:
            del configs[board_info]

    if print_info:
        print("Writing data to new file.")

    if data_writer is None:
        with open(output_filenames[0], 'wb') as writer:
            pickle.dump(configs, writer, pickle.HIGHEST_PROTOCOL)
    else:
        data_writer(configs, output_filenames)



def during_search_n_man_filter_creator(n):
    """
    Creates and returns a filter function for use in the create_database_from_pgn function, to be used during data acquisition.
    The function filters out boards which have less than or equal to a given number of chess pieces (n).
    """
    def filter(board, move):
        if popcount(board.occupied) <= n:
            return True
        return False

    return filter


def standard_comparison_move_generator(board, data):
    """
    A move generator to produce moves such that from the given board, making one of the moves would result in
    the random board from the (old, new, random) triplet used with loss functions similar to that of Deep Pink.
    This function is titled standard because all it does is generate every move which was not chosen by a player
    in any game in the databases used.
    """
    for move in generate_legal_moves(board, BB_ALL, BB_ALL):
        if data.moves.get((move.from_square, move.to_square,None if move.promotion== 0 else move.promotion)) is None:
            yield move



def board_eval_data_writer_creator(file_ratios, for_deep_pink_loss, comparison_move_generator=None,
                        print_frequency=10000):
    """
    Creates a data writer for use in the create_database_from_pgn function.  It can create a set of database files
    containing both files that can be used with Deep Pink's loss function (or one based on a similar idea),
    and files containing a pickled version of the dictionary given (for outside use).  Though this is intended for use
    with the create_database_from_pgn function, it can very easily be used on already pickled dictionaries.  So it often
    is preferable to pickle the dictionaries first, so that multiple iterations of databases can be created from them
    without having to go through create_database_from_pgn again.

    :param file_ratios: The ratios of the given boards to use when creating each database file.
    :param for_deep_pink_loss: An array of boolean values indicating which of the database files being created should be
     formatted for use with Deep Pink's loss function (or a loss function with a similar idea).
    :param comparison_move_generator: A generator function taking a BoardState and BoardData object as parameters.
     The generator produces moves such that from the given board, making one of the moves would result in
     the random board in the (old, new, random) triplet.
    :param print_frequency: The increment in which to print the number of boards processed and written to file,
     and how much time that took.
    """



    if comparison_move_generator is None:
        comparison_move_generator = standard_comparison_move_generator



    def writer_fn(dict, filenames):
        """
        The function to be returned by board_eval_data_writer_creator.
        """

        def create_serialized_example(array_to_write):
            """
            NOTES:
            1) I should be doing this like it's done in the move generation database generation, where a board is a
            set of 64 uint8s.
            """
            return tf.train.Example(features=tf.train.Features(feature={
                "boards": tf.train.Feature(int64_list=tf.train.Int64List(value=array_to_write))})).SerializeToString()


        writers = [tf.python_io.TFRecordWriter(file) if for_deep_pink else open(file, 'wb') for file, for_deep_pink in
                   zip(filenames, for_deep_pink_loss)]

        number_of_boards = len(dict)

        if not print_frequency is None:
            print("Number of board configurations:", number_of_boards)
        start_time = time.time()

        cur_entry_num = 0

        dict_iterator = iter(dict.items())

        for writer, ratio, should_get_deep_pink_loss in zip(writers, file_ratios, for_deep_pink_loss):
            if not should_get_deep_pink_loss:
                pickle.dump({next(dict_iterator) for _ in range(int(math.floor(ratio * number_of_boards)))}, writer, pickle.HIGHEST_PROTOCOL)
                cur_entry_num += int(math.floor(ratio * number_of_boards))
                if not print_frequency is None:
                    print(cur_entry_num, "boards writen (just completed a file's pickle.dump)")
            else:
                for _ in range(int(math.floor(ratio * number_of_boards))):

                    if not print_frequency is None and cur_entry_num % print_frequency == 0:
                        print(cur_entry_num, "total boards writen.  The time since the previous print:",
                              time.time() - start_time)
                        start_time = time.time()

                    cur_board_data, cur_data = next(dict_iterator)

                    temp_board = board_state_from_dict_key(cur_board_data)

                    original_board_data_to_write = get_feature_array(cur_board_data)

                    # Get the maximum number of times a move was chosen from the current position
                    # most_chosen_move= max(cur_data.moves, key=cur_data.moves.get)

                    max_move_count = max(cur_data.moves.values())

                    # Get the set of moves chosen max_move_count number of times
                    most_chosen_moves = [move for move in cur_data.moves.keys() if cur_data.moves.get(move) == max_move_count]

                    most_chosen_move_picked = most_chosen_moves[random.randrange(len(most_chosen_moves))]


                    most_chosen_data = get_feature_array(
                            flip_vertically(
                                get_board_info_tuple(
                                    copy_push(
                                        temp_board,
                                        Move(
                                            most_chosen_move_picked[0],
                                            most_chosen_move_picked[1],
                                            0 if most_chosen_move_picked[2] is None else most_chosen_move_picked[2])))))

                    comparison_moves = [move for move in comparison_move_generator(temp_board, cur_data)]
                    if len(comparison_moves) != 0:
                        comparison_move = comparison_moves[random.randrange(len(comparison_moves))]
                        for_comparison_board = get_feature_array(
                                    flip_vertically(
                                        get_board_info_tuple(
                                            copy_push(temp_board, comparison_move))))


                        writer.write(create_serialized_example(np.concatenate([original_board_data_to_write, most_chosen_data, for_comparison_board])))

                        # example_triplets = product([original_board_data_to_write], most_chosen_data, for_comparison_boards)

                        # serialized_data_strings = list(map(lambda x:create_serialized_example(np.concatenate(x)), example_triplets))

                        # for cur_string in serialized_data_strings:
                        #     writer.write(cur_string)

                        cur_entry_num += 1

            writer.close()

    return writer_fn


@nb.njit
def jitted_set_up_struct_moves(struct_array):
    for j in range(len(struct_array)):
        set_up_move_array(struct_array[j])




def generate_tf_records_with_child_values(node_batch_eval_fn, print_interval=100, batch_size=200, num_workers=None):
    """
    A function to generate a TFRecords dataset for the training of the move scoring neural network.  It is designed in
    a way such that it could be given to create_database_from_pgn as it's data_writer.

    :param node_batch_eval_fn: A function to score a set of GameNode objects
    :param print_interval: The interval in which to print the functions progress and time taken since last print
    :param batch_size: The number of examples to be prepared at a time by each thread
    :param num_workers: The number of workers to be created/used by the ThreadPoolExecutor (used when generating
     batches of data to write)

    NOTES:
    1) If a board has more than the amount of moves that a board_info scalar can hold it will cause an error
    """
    def create_serialized_example(white_info, moves, results):
        example = tf.train.Example(features=tf.train.Features(feature={
            "board" : tf.train.Feature(int64_list=tf.train.Int64List(value=get_feature_array(white_info))),
            "from_squares": tf.train.Feature(int64_list=tf.train.Int64List(value=moves[:,0])),
            "to_squares": tf.train.Feature(int64_list=tf.train.Int64List(value=moves[:,1])),
            "move_scores" : tf.train.Feature(float_list=tf.train.FloatList(value=results)),
            "num_moves" : tf.train.Feature(int64_list=tf.train.Int64List(value=[len(moves)])),
        }))

        return example.SerializeToString()

    def keys_to_serialized_example(dict_keys):
        boards = board_info_array_from_dict_keys(dict_keys)

        jitted_set_up_struct_moves(boards)

        lengths = boards['children_left']

        moves_lists = list(map(lambda b: b['unexplored_moves'][:b['children_left']], boards))

        moves = np.concatenate(moves_lists)

        children = np.repeat(boards, lengths)

        push_moves(children, moves)

        evaluated_results = node_batch_eval_fn(children)

        result_splitters = np.r_[0, np.cumsum(lengths,dtype=np.int32)]

        the_results = list(evaluated_results[start:end] for start, end in zip(result_splitters[:-1],result_splitters[1:]))

        return list(map(create_serialized_example, dict_keys, moves_lists, the_results))


    def the_writer(the_dict, output_filenames):
        """
        :param output_filenames: An array of size 1 containing the desired output filename/location of
         of the resulting TFRecords database.   (This is an array of size one to adhere to the basic structure of the
         data_writer described in the create_database_from_pgn function)
        """
        writer = tf.python_io.TFRecordWriter(output_filenames[0])
        cur_keys = []
        start_time = time.time()
        futures = []
        with ThreadPoolExecutor(num_workers) as executor:
            for index, key in enumerate(the_dict):
                cur_keys.append(key)


                if index % batch_size == 0 and index != 0:
                    futures.append(executor.submit(keys_to_serialized_example, cur_keys))
                    cur_keys = []

            print("Tasks have been submitted in time:", time.time() - start_time)
            start_time = time.time()
            for index, cur_future in enumerate(as_completed(futures)):
                for cur_str in cur_future.result():
                    writer.write(cur_str)

                if index % print_interval == 0 and index != 0:
                    print(index*batch_size, "boards processed, the last", print_interval*batch_size,"in time:", time.time() - start_time, "seconds")
                    start_time = time.time()

        writer.close()


    return the_writer


def known_scoring_writer_creator(score_board, print_info=True):
    """
    :param score_board: A function accepting a Python-Chess Board and returning an object with attributes cp (score)
    and mate (depth to mate, with negative's being losses).  All of this is relative to the white player.
    """
    def create_serialized_example(white_info, score):
        example = tf.train.Example(features=tf.train.Features(feature={
            "board" : tf.train.Feature(int64_list=tf.train.Int64List(value=get_feature_array(white_info))),
            "score" : tf.train.Feature(int64_list=tf.train.Int64List(value=[score])),
        }))

        return example.SerializeToString()


    def writer(the_dict, output_filenames):
        """
        :param output_filenames: must be length 1
        """
        with tf.python_io.TFRecordWriter(output_filenames[0]) as the_writer:
            py_boards = map(dict_key_to_py_board, the_dict)
            for index, (py_board, key) in enumerate(zip(py_boards, the_dict)):
                if print_info and index % 10000== 0:
                    print("Completed %d boards so far."%index)

                scoring_results = score_board(py_board)
                if scoring_results.cp is None:
                    # If a mate is found, store the win/loss value,
                    # scaled such that the magnitude of the win/loss value will decrease  as the depth from the mate
                    # increases.

                    mate_depth = scoring_results.mate
                    mate_value = np.iinfo(np.int64).max if mate_depth > 0 else np.iinfo(np.int64).min
                    score_to_write = mate_value - mate_depth
                else:
                    score_to_write = scoring_results.cp

                the_writer.write(create_serialized_example(key, score_to_write))

    return writer



def board_info_numpy_file_writer(the_dict, output_filenames):
    """
    :param output_filenames: MUST BE LENGTH ONE
    """
    struct_array = board_info_array_from_dict_keys(the_dict.keys())
    print("Completed BoardInfo struct creation")
    for j in range(len(struct_array)):
        if j % 100000== 0:
            print("Generated %d move's so far."%j)

        set_up_move_array(struct_array[j])

    np.save(output_filenames[0], struct_array)







def stockfish_move_generator_creator(sf_location, sf_threads, sf_time):

    stockfish_ai = chess.uci.popen_engine(sf_location)
    stockfish_ai.setoption({"threads" : sf_threads})


    def comparison_fn(board, data):
        py_board = chess.Board()

        py_board.kings = board.kings
        py_board.queens = board.queens
        py_board.rooks = board.rooks
        py_board.bishops = board.bishops
        py_board.knights = board.knights
        py_board.pawns = board.pawns

        py_board.occupied = board.occupied
        py_board.occupied_co[True] = board.occupied_w
        py_board.occupied_co[False] = board.occupied_b

        py_board.castling_rights = board.castling_rights

        py_board.ep_square = board.ep_square

        py_board.turn = board.turn

        move_wasnt_in_computed_data_fn = lambda m : data.moves.get((m.from_square, m.to_square, None if m.promotion == 0 else m.promotion)) is None

        moves_to_search = [move for move in py_board.generate_legal_moves() if move_wasnt_in_computed_data_fn(move)]

        stockfish_ai.position(py_board)

        chosen_move = stockfish_ai.go(searchmoves=moves_to_search, movetime=sf_time).bestmove

        yield Move(chosen_move.from_square, chosen_move.to_square, 0 if chosen_move.promotion is None else chosen_move.promotion)


    return comparison_fn



def game_length_filter_creator(max_ply):
    def filter(game):
        if int(game.headers['PlyCount'])  > max_ply:
            return True
        return False

    return filter


def excessive_promotion_filter_creator(max_promotions):
    def filter(game):
        num_promotions = 0
        for move in game.main_line():
            if not move.promotion is None:
                if num_promotions >= max_promotions:
                    return True
                num_promotions += 1
        return False

    return filter



def create_sf_scorer(sf_location, sf_time=15, sf_threads=1):
        stockfish_ai = chess.uci.popen_engine(sf_location)
        stockfish_ai.setoption({"threads": sf_threads})

        info_handler = chess.uci.InfoHandler()
        stockfish_ai.info_handlers.append(info_handler)

        def score_board(py_board):
            stockfish_ai.position(py_board)
            stockfish_ai.go(movetime=sf_time)

            return info_handler.info["score"][1]

        return score_board



if __name__ == "__main__":

    """
    The code below is a sample of how these functions are used to generate the databases used by this engine.     
    """




    pgn_file_nums = [
        "1505782","1505781","1505780","1505779","1505778","1505777","1490172","1490171","1490170",
        "1490169","1490168","1490167","1490166","1490165","1490164","1490163","1490162","1490161"]


    pgn_file_paths = list(map(lambda year, num : "/srv/databases/from_pycharm/fics/ficsgamesdb_%d_standard2000_nomovetimes_%s.pgn"%(year, num), range(1999,2017), pgn_file_nums))

    pickle_filenames = ["encoder_training_set_%d.pkl"%j for j in range(9)] + \
                       ["encoder_validation_set_%d.pkl" % j for j in range(2)] + \
                       ["encoder_testing_set_1.pkl"] + \
                       ["scoring_training_set_%d.pkl" % j for j in range(9)] + \
                       ["scoring_validation_set_%d.pkl" % j for j in range(3)] + \
                       ["scoring_testing_set_0.pkl"] + \
                       ["move_scoring_training_set_%d.pkl" % j for j in range(9)] + \
                       ["move_scoring_validation_set_%d.pkl" % j for j in range(2)] + \
                       ["move_scoring_testing_set_0.pkl"]


    final_dataset_filenames = list(map(lambda x: "/srv/databases/chess_engine/full_10/" + x, pickle_filenames))

    temp_dataset_filenames = list(map(lambda x: "/srv/databases/chess_engine/sf_chosen_moves_4/" + x, pickle_filenames))
    temp_dataset_filenames2 = list(map(lambda x: "/srv/databases/chess_engine/sf_chosen_moves_6/" + x, pickle_filenames))
    temp_dataset_filenames3 = list(map(lambda x: "/srv/databases/chess_engine/move_scoring_2/" + x, pickle_filenames))


    file_ratios = [.025] * len(final_dataset_filenames)
    for_deep_pink_loss_use = [False]*len(final_dataset_filenames)


    # Create the databases for use in training the board evaluation neural network.  Long term this will likely be combined
    # with the generation of the move scoring database to prevent any overlap in boards between the datasets.
    start_time = time.time()
    create_database_from_pgn(
        pgn_file_paths[:-1],
        game_filters=[game_length_filter_creator(170),
                      excessive_promotion_filter_creator(4)],
        # to_collect_filters=[during_search_n_man_filter_creator(3)],
        data_writer=board_eval_data_writer_creator(
            file_ratios,
            for_deep_pink_loss_use,
            comparison_move_generator=standard_comparison_move_generator,
            print_frequency=10000),
        # num_first_moves_to_skip=4,
        output_filenames=temp_dataset_filenames)#=#final_dataset_filenames)

    print("Time taken to create databases:", time.time() - start_time)



    # BOARD_EVAL_GRAPHDEF_FILENAME = "/srv/tmp/encoder_evaluation_helper/no_modules_5/1532030632/tensorrt_eval_graph.pb"#"/srv/tmp/encoder_evaluation_helper/sf_data_attempts/encoder_new_data_2.6/1529409998/tensorrt_eval_graph.pb"
    # BOARD_PREDICTOR, _, PREDICTOR_CLOSER = get_inference_functions(BOARD_EVAL_GRAPHDEF_FILENAME, None, session_gpu_memory=.15)
    #
    #
    # def cur_eval_fn(struct_array):
    #     return BOARD_PREDICTOR(
    #         *struct_array_to_ann_inputs(
    #             struct_array,
    #             np.array([], dtype=numpy_node_info_dtype),
    #             np.ones(len(struct_array), dtype=np.bool_),
    #             np.array([], dtype=np.bool_),
    #             len(struct_array))).squeeze(axis=1)


    STOCKFISH_LOCATION = "/home/sam/PycharmProjects/ChessAI/stockfish-8-linux/Linux/stockfish_8_x64"


    # board_eval_writer = board_eval_data_writer_creator(
    #         [1],
    #         [True],
    #         comparison_move_generator=stockfish_move_generator_creator(STOCKFISH_LOCATION, 1, 1), #Using very little resources so that it won't find a better move than the GM had
    #         print_frequency=1000)



    sf_scoring_writer = known_scoring_writer_creator(
        create_sf_scorer(
            sf_location=STOCKFISH_LOCATION,
            sf_time=100,
            sf_threads=3))


    # VERY IMPORTANT NOTE:
    # I'm not sure the exact reason, but for some reason using multiple workers in the below data writer
    # will caused strange errors which seem to stem from something to do with TensorRT.
    # move_scoring_writer = generate_tf_records_with_child_values(cur_eval_fn, num_workers=1, print_interval=100,batch_size=50)

    file_index = 25
    print("Creating database for file",file_index)
    INPUT_FILENAME = temp_dataset_filenames2[file_index]
    OUTPUT_FILENAME = temp_dataset_filenames3[file_index][:-3] + "tfrecords" #"npy"

    with open(INPUT_FILENAME, "rb") as input:
        # cur_dict_to_write = {cur_tuple[0]: cur_tuple[1] for cur_tuple in pickle.load(input)}
        cur_set_to_write = {cur_tuple[0] for cur_tuple in pickle.load(input)}


    sf_scoring_writer(cur_set_to_write, [OUTPUT_FILENAME])

    # board_eval_writer(cur_dict_to_write, [OUTPUT_FILENAME])

    # move_scoring_writer(cur_set_to_write, [OUTPUT_FILENAME])

    # board_info_numpy_file_writer(cur_dict_to_write ,[OUTPUT_FILENAME])

    # PREDICTOR_CLOSER()


