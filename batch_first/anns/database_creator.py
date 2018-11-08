import tensorflow as tf

import re
import pickle
import time
import chess.pgn

from batch_first.numba_board import *




def game_iterator(pgn_filename):
    """
    Iterates through the games stored on the given pgn file (as python-chess Game objects).
    """
    with open(pgn_filename) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            yield game


def get_gamenode_after_moves(game, num_moves):
    """
    Get the node from a given game after a specified number of moves has been made.
    """
    for _ in range(num_moves):
        game = game.variations[0]
    return game


def get_py_board_info_tuple(board):
    return np.array([board.kings, board.queens, board.rooks, board.bishops, board.knights, board.pawns,
                     board.castling_rights, 0 if board.ep_square is None else BB_SQUARES[np.int32(board.ep_square)],
                     board.occupied_co[board.turn], board.occupied_co[not board.turn], board.occupied], dtype=np.uint64)


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


def additional_move_features(board_features):
    return {"move_from_square": tf.train.Feature(int64_list=tf.train.Int64List(value=[board_features[1]])),
            "move_to_square": tf.train.Feature(int64_list=tf.train.Int64List(value=[board_features[2]])),
            "move_filter": tf.train.Feature(int64_list=tf.train.Int64List(value=[board_features[3]]))}


def serializer_creator(additional_feature_dict_fn=None):
    """
    A convenience function to aid in the use of the combine_pickles_and_create_tfrecords function.

    :param additional_feature_dict_fn: A function which returns a dictionary mapping feature names (strings)
     to tf.train.Feature objects.  The features given in this dict will be serialized along with the "board" and
     "score" features
    :return: A function to be given to the combine_pickles_and_create_tfrecords function as
     the serializer parameter
    """
    def serializer_to_return(board_features, score):
        feature_input = board_features[0] if isinstance(board_features[0], tuple) else board_features
        feature_dict = {
            "board": tf.train.Feature(int64_list=tf.train.Int64List(value=get_feature_array(feature_input))),
            "score": tf.train.Feature(int64_list=tf.train.Int64List(value=[score]))}

        if additional_feature_dict_fn:
            feature_dict.update(additional_feature_dict_fn(board_features))

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example.SerializeToString()

    return serializer_to_return


def get_nodes_value(node, board_turn, max_win_value=1000000):
    """
    Gets the value of the given node, or None if no value is stored.

    :param node: The node who's value is to be returned
    :param board_turn: If this parameter is False, it will multiply the score by -1 (changing the score to be with
     respect to the other player)
    :param win_value: The value for a depth-0 winning board.  The value of a depth-1 win would thus be one less,
     and depth-n would be n less
    """
    re_search_results = re.search(r'\[\%eval (.*?)\]', node.comment)
    if re_search_results is None:
        return None

    sf_score_str = re_search_results.group(1)

    # If a mate is found, store the win/loss value,
    # scaled such that the magnitude of the win/loss value will decrease as the depth from the mate
    # increases.
    if sf_score_str[0] == '#':
        mate_depth = np.int64(int(sf_score_str[1:]))

        mate_value = max_win_value if mate_depth > 0 else -max_win_value

        sf_score = mate_value - mate_depth
    else:
        sf_score = np.int64(float(sf_score_str) * 100)  # converts the score to the traditional centi-pawn scores used by StockFish

    if not board_turn:
        sf_score *= -1

    return sf_score


def get_data_from_pgns(pgn_filenames, output_filename, get_board_for_game_fn, print_interval=10000):
    """
    Creates a pickle database from the data in the given pgn files.  One example is produced per game, and is done so
    by a given function.


    :param pgn_filenames: An iterable of the names/paths of pgn files to have data gathered from
    :param output_filename: A string used as the name of the output pickle database
    :param get_board_for_game_fn: A function that accepts a python-chess Game, and returns a length two tuple.  That tuple's
     first element is either the board's representation (a tuple), or a tuple with it's representation as it's first
     value (followed by any other desired information) (All of this information is used to ensure that each example is unique).
     The second element of the returned tuple is the score associated with the example
    :param print_interval: The number of games checked between printing the progress
    """
    eval_boards = {}
    num_checked = 0
    for j, filename in enumerate(pgn_filenames):
        if print_interval:
            print("Starting file %d"%j)
        last_print = time.time()

        for returns in map(get_board_for_game_fn, game_iterator(filename)):
            if not returns is None:
                if eval_boards.get(returns[0]) is None:
                    eval_boards[returns[0]] = returns[1]

            num_checked += 1

            if print_interval and num_checked % print_interval == 0:
                print("%d boards have been checked, generating %d usable boards, with %f time since the last print"%(num_checked, len(eval_boards), time.time() - last_print))
                last_print = time.time()

    with open(output_filename, 'wb') as writer:
        pickle.dump(eval_boards, writer, pickle.HIGHEST_PROTOCOL)


def combine_pickles_and_create_tfrecords(filenames, output_filenames, output_ratios, serializer):
    """
    Combines the information in the given pickle files (maintaining uniqueness), serializes the data, then saves it
    as a set of tfrecords files (in the desired ratios).

    :param filenames: An iterable of pickled filenames (produced by get_data_from_pgns) to serialize and combine
    :param output_filenames: An iterable of filenames to save the serialized tfrecords to
    :param output_ratios: The ratios of the combined data to be saved to each of the files in output_filenames
    :param serializer: A function which returns a serialized TensorFlow Example, the details of it's two parameters
     are described in the get_data_from_pgns's comments as the return of the get_board_for_game_fn parameter
     (the serializer_creator function can be used to generate this parameter)
    """
    combined_dict = {}
    for name in filenames:
        with open(name, "rb") as to_read:
            for key, value in pickle.load(to_read).items():
                if combined_dict.get(key) is None:
                    combined_dict[key] = value

    break_indices = np.r_[0, (len(combined_dict) * np.cumsum(np.array(output_ratios))[:-1]).astype(np.int32),len(combined_dict)]
    dict_iterator = iter(combined_dict)

    key_arrays = ([next(dict_iterator) for _ in range(j-i)] for i,j in zip(break_indices[:-1],break_indices[1:]))
    serialized_examples = (map(serializer, keys, (combined_dict[k] for k in keys)) for keys in key_arrays)

    for filename, examples in zip(output_filenames, serialized_examples):
        with tf.python_io.TFRecordWriter(filename) as writer:
            for example in examples:
                writer.write(example)


def create_board_eval_board_from_game_fn(min_start_moves=6, for_testing=False):
    random_number_starter = min_start_moves - 1
    min_boards_in_game = min_start_moves + 1
    def get_board_for_game(game):
        num_boards_in_game = len(list(game.main_line()))

        if num_boards_in_game <= min_boards_in_game:
            return None

        desired_game_node = get_gamenode_after_moves(game, np.random.randint(random_number_starter, num_boards_in_game-1))
        py_board = desired_game_node.board()

        if for_testing:
            sf_score = 0
        else:
            sf_score = get_nodes_value(desired_game_node,  py_board.turn)
            if sf_score is None:
                return None

        if py_board.turn:
            white_info = tuple(get_py_board_info_tuple(py_board))
        else:
            white_info = tuple(flip_vertically(get_py_board_info_tuple(py_board)))

        return white_info, sf_score
    return get_board_for_game


def create_move_scoring_board_from_game_fn(min_start_moves=6):
    random_number_starter = min_start_moves - 1
    min_boards_in_game = min_start_moves + 1
    def get_board_for_game(game):
        num_boards_in_game = len(list(game.main_line()))

        if num_boards_in_game <= min_boards_in_game:
            return None

        desired_game_node = get_gamenode_after_moves(game, np.random.randint(random_number_starter, num_boards_in_game-1))

        py_board = desired_game_node.board()

        next_node = desired_game_node.variations[0]
        next_value = get_nodes_value(next_node, py_board.turn)

        if next_value is None:
            return None

        # Store the move and flip it if the board was converted from black's perspective
        move_made = next_node.move
        if not py_board.turn:
            move_made.from_square = chess.square_mirror(move_made.from_square)
            move_made.to_square = chess.square_mirror(move_made.to_square)

        if move_made.promotion is None or move_made.promotion == chess.QUEEN:
            move_promotion = NO_PROMOTION_VALUE
        else:
            move_promotion = move_made.promotion

        move_filter = MOVE_FILTER_LOOKUP[
            move_made.from_square,
            move_made.to_square,
            move_promotion]

        if py_board.turn:
            white_info = tuple(get_py_board_info_tuple(py_board))
        else:
            white_info = tuple(flip_vertically(get_py_board_info_tuple(py_board)))

        return (white_info, move_made.from_square, move_made.to_square, move_filter), next_value

    return get_board_for_game


def save_all_boards_from_game_as_npy(pgn_file, output_filename, max_games=100000, print_interval=5000):
    """
    Goes through the games in a pgn file and saves the unique boards in NumPy file format
    (with dtype numpy_node_info_dtype).  Prior to being saved the legal move arrays are set up.

    :param pgn_file: The pgn file to gather boards from
    :param output_filename: The name for the database file to be created
    :param max_games: The maximum number of games to go through
    :param print_interval: The number of games between each progress update

    NOTES:
    1) This function uses a large amount of memory (mainly caused by np.unique)
    """
    prev_time = time.time()
    root_struct = create_node_info_from_python_chess_board(chess.Board())

    collected = []
    for j, game in enumerate(game_iterator(pgn_file)):

        if j == max_games:
            break

        if j % print_interval == 0:
            print("%d games complete with %d boards collected (not unique) with %f time since last print."%(j, len(collected), time.time() - prev_time))
            prev_time = time.time()

        struct = root_struct.copy()

        move_iterator = (np.array([[move.from_square, move.to_square, 0 if move.promotion is None else move.promotion]]) for move in game.main_line())
        for move_ary in move_iterator:
            push_moves(struct, move_ary)
            collected.append(struct.copy())

    print("Completed board acquisition")

    unique_structs = np.unique(np.array(collected))

    print("%d unique boards produced." % len(unique_structs))

    set_up_move_arrays(unique_structs)

    structs_with_less_than_max_moves = unique_structs[unique_structs['children_left'] <= MAX_MOVES_LOOKED_AT]

    print("Moves have now been set up.")

    np.save(output_filename, structs_with_less_than_max_moves)


def get_zero_valued_boards(filename, output_filename, print_interval=25000):
    def iterate_zero_value_nodes():
        num_done = 0
        for game in game_iterator(filename):
            while len(game.variations) == 1:
                game = game.variations[0]
                if get_nodes_value(game, True) == 0:
                    num_done += 1
                    if num_done % print_interval == 0:
                        print("Zero valued boards gathered:", num_done)
                    yield game.board()


    to_return = np.array([create_node_info_from_python_chess_board(b) for b in iterate_zero_value_nodes()])
    unique_boards = np.unique(to_return)

    np.save(output_filename, unique_boards)


def get_locations_of_lines_that_pass_filters(filename, filters, print_interval=1e7):
    """
    Gets the line numbers of the lines in a given file that pass a set of filters.
    """
    def passes(line):
        for filter in filters:
            if filter(line):
                return False
        return True

    with open(filename, 'r') as f:
        prev_time = time.time()
        line_nums = []
        for j,line in enumerate(iter(f.readline, '')):
            if passes(line):
                line_nums.append(f.tell())

            if j % print_interval == 0:
                print("%d lines completed so far with %d lines found in %f time since last print"%(j, len(line_nums), time.time() - prev_time))
                prev_time = time.time()

        return line_nums


def clean_pgn_file(pgn_to_filter, output_filename, line_filters=[], header_filters=[]):
    def passes_header_filters(headers):
        for filter in header_filters:
            if filter(headers):
                return False
        return True

    line_nums = get_locations_of_lines_that_pass_filters(
        pgn_to_filter,
        line_filters)
    line_nums = line_nums[: -1]  #the last game isn't used because if it's the last game in the file the array indexing to follow will cause an error

    line_nums = np.array(line_nums)
    print("Desired line numbers have been found.")
    with open(output_filename, 'w') as writer:
        with open(pgn_to_filter, 'r') as f:
            temp_stuff = [(offset, passes_header_filters(header)) for offset, header in chess.pgn.scan_headers(f)]

            offsets, should_write = zip(*temp_stuff)

            should_write = np.array(list(should_write), dtype=np.bool_)

            game_offsets = np.array(list(offsets))

            print("Game offsets calculated and headers collected")

            temp_indices = np.searchsorted(game_offsets, line_nums)


            actual_offset_indices = temp_indices - 1

            should_write = should_write[actual_offset_indices]

            amount_to_write = game_offsets[temp_indices] - game_offsets[actual_offset_indices]
            filtered_start_lines = game_offsets[actual_offset_indices]

            print("Starting to write the new file.")

            for offset, amount in zip(filtered_start_lines[should_write], amount_to_write[should_write]):
                f.seek(offset)
                writer.write(f.read(amount))


def combine_numpy_files_and_make_unique(filenames, output_name):
    combined = np.concatenate([np.load(file) for file in filenames])
    unique_boards = np.unique(combined)
    np.save(output_name, unique_boards)


if __name__ == "__main__":
    """
    The commented out code throughout the rest of this file is how the ANN training databases are created,
    as well as any other database used in Batch First. 
    """


    def add_path_fn_creator(path):
        return lambda x : [path + y for y in x]

    PGN_FILENAMES_WITHOUT_PATHS = [
        "lichess_db_standard_rated_2018-06.pgn",
        "lichess_db_standard_rated_2018-05.pgn",
        "lichess_db_standard_rated_2018-04.pgn",
        "lichess_db_standard_rated_2018-03.pgn",
        "lichess_db_standard_rated_2018-02.pgn",
        "lichess_db_standard_rated_2018-01.pgn",
        "lichess_db_standard_rated_2017-12.pgn",
        "lichess_db_standard_rated_2017-11.pgn",
        "lichess_db_standard_rated_2017-10.pgn",
        "lichess_db_standard_rated_2017-09.pgn",
        "lichess_db_standard_rated_2017-08.pgn",
        "lichess_db_standard_rated_2017-07.pgn",
        "lichess_db_standard_rated_2017-06.pgn",
        "lichess_db_standard_rated_2017-05.pgn",
        "lichess_db_standard_rated_2017-04.pgn",
        "lichess_db_standard_rated_2017-03.pgn",
        "lichess_db_standard_rated_2017-02.pgn",
        "lichess_db_standard_rated_2017-01.pgn",
        "lichess_db_standard_rated_2016-12.pgn",
        "lichess_db_standard_rated_2016-11.pgn",
        "lichess_db_standard_rated_2016-10.pgn",
        "lichess_db_standard_rated_2016-09.pgn",
        "lichess_db_standard_rated_2016-08.pgn",
        "lichess_db_standard_rated_2016-07.pgn",
        "lichess_db_standard_rated_2016-06.pgn",
        "lichess_db_standard_rated_2016-05.pgn",
        "lichess_db_standard_rated_2016-04.pgn",
        "lichess_db_standard_rated_2016-03.pgn",
        "lichess_db_standard_rated_2016-02.pgn",
        "lichess_db_standard_rated_2016-01.pgn",][::-1]

    PICKLE_FILENAMES = list(map(lambda s : s[:-3] + "pkl", PGN_FILENAMES_WITHOUT_PATHS))

    PGN_FILENAMES = add_path_fn_creator("/srv/databases/lichess/original_pgns/")(PGN_FILENAMES_WITHOUT_PATHS)

    FILTERED_FILENAMES = add_path_fn_creator("/srv/databases/lichess/filtered_pgns/")(PGN_FILENAMES_WITHOUT_PATHS)
    FILTERED_FILENAMES = list(map(lambda f: "%s_filtered.pgn" % f[:-4], FILTERED_FILENAMES))


    TFRECORDS_FILENAMES = [
        "lichess_training.tfrecords",
        "lichess_validation.tfrecords",
        "lichess_testing.tfrecords"]

    TFRECORDS_OUTPUT_RATIOS = [.85, .1, .05]






    pgn_not_used_for_ann_training = "/srv/databases/lichess/lichess_db_standard_rated_2018-07.pgn"
    npy_output_filename = "/srv/databases/lichess/lichess_db_standard_rated_2018-07_first_100k_games"
    # save_all_boards_from_game_as_npy(pgn_not_used_for_ann_training, npy_output_filename)


    file_index = -6
    OUTPUT_FILTERED_FILENAME = add_path_fn_creator("/srv/databases/has_zero_valued_board/")(PGN_FILENAMES_WITHOUT_PATHS)[file_index]
    # clean_pgn_file(
    #     FILTERED_FILENAMES[file_index],
    #     OUTPUT_FILTERED_FILENAME,
    #     [lambda s: s[0] != '1',
    #      lambda s: not "%eval 0.0" in s],
    #     [lambda h: h['Termination'] != "Normal"])


    file_index = -2
    INPUT_FILENAME = FILTERED_FILENAMES[file_index]
    OUTPUT_FILENAME = INPUT_FILENAME[:-4] + "_zero_boards"
    # get_zero_valued_boards(INPUT_FILENAME, OUTPUT_FILENAME)


    ########################The commented out code below is used for the move scoring database###################
    file_index = 6
    OUTPUT_FILENAME = add_path_fn_creator("/srv/databases/lichess_just_move_scoring_fixed_ag_promotion/")(PGN_FILENAMES_WITHOUT_PATHS)[file_index][:-3] + "pkl"
    OUTPUT_FILENAME = OUTPUT_FILENAME[:-4] + "_second_pass.pkl"

    CUR_MOVE_PGN_FILENAME = FILTERED_FILENAMES[file_index]
    # get_data_from_pgns(
    #     [CUR_MOVE_PGN_FILENAME],
    #     OUTPUT_FILENAME,
    #     create_move_scoring_board_from_game_fn(5))


    #########################The commented out code below is used for the board evaluation database###################
    file_index = 6
    CUR_EVAL_PGN_FILENAME = FILTERED_FILENAMES[file_index]
    OUTPUT_FILENAME = add_path_fn_creator("/srv/databases/lichess_combined_methods_eval_databases/")(PGN_FILENAMES_WITHOUT_PATHS)[file_index][:-3] + "pkl"
    # OUTPUT_FILENAME = OUTPUT_FILENAME[:-4] + "_second_pass.pkl"
    # get_data_from_pgns(
    #     [CUR_EVAL_PGN_FILENAME],
    #     OUTPUT_FILENAME,
    #     create_board_eval_board_from_game_fn())



    #########################The commented out code below is used for the board evaluation database#########################
    eval_path_adder = add_path_fn_creator("/srv/databases/lichess_combined_methods_eval_databases/")

    EVAL_PICKLE_FILES = eval_path_adder(PICKLE_FILENAMES)
    # EVAL_PICKLE_FILES += list(map(lambda s: s[:-4] + "_second_pass.pkl", EVAL_PICKLE_FILES))

    EVAL_OUTPUT_TFRECORDS_FILES = eval_path_adder(TFRECORDS_FILENAMES)

    # combine_pickles_and_create_tfrecords(
    #     EVAL_PICKLE_FILES,
    #     EVAL_OUTPUT_TFRECORDS_FILES,
    #     TFRECORDS_OUTPUT_RATIOS,
    #     serializer_creator())


    #########################The commented out code below is used for the move scoring database###################
    policy_path_adder = add_path_fn_creator("/srv/databases/lichess_just_move_scoring_fixed_ag_promotion/")
    MOVE_SCORING_PICKLE_FILES = policy_path_adder(PICKLE_FILENAMES)
    # MOVE_SCORING_PICKLE_FILES += list(map(lambda s: s[:-4] + "_second_pass.pkl", MOVE_SCORING_PICKLE_FILES))
    POLICY_OUTPUT_TFRECORDS_FILES = policy_path_adder(TFRECORDS_FILENAMES)

    # combine_pickles_and_create_tfrecords(
    #     MOVE_SCORING_PICKLE_FILES,
    #     POLICY_OUTPUT_TFRECORDS_FILES,
    #     TFRECORDS_OUTPUT_RATIOS,
    #     serializer_creator(additional_move_features))

