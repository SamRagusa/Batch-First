import tensorflow as tf

import re
import pickle
import time

from batch_first.numba_board import *




def generate_move_filter_table():
    """
    Generate a lookup table for the policy encoding described in the following paper:

    'Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm'
    https://arxiv.org/pdf/1712.01815.pdf

    So in the returned table, the value at index (f,t,p) is the index of the policy plane (in the move scoring ann)
    associated with the chess move described as moving a piece from square f to square t and being promoted to piece p.
    """
    diffs = {}
    for j in range(1, 8):
        diffs[(0, j)] = j - 1
        diffs[(0, -j)] = j + 6
        diffs[(j, 0)] = j + 13
        diffs[(-j, 0)] = j + 20
        diffs[(j, j)] = j + 27
        diffs[(j, -j)] = j + 34
        diffs[(-j, j)] = j + 41
        diffs[(-j, -j)] = j + 48

    diffs[(1, 2)] = 56
    diffs[(1, -2)] = 57
    diffs[(-1, 2)] = 58
    diffs[(-1, -2)] = 59
    diffs[(2, 1)] = 60
    diffs[(2, -1)] = 61
    diffs[(-2, 1)] = 62
    diffs[(-2, -1)] = 63

    filter_table = np.zeros([64,64,6], dtype=np.uint8)

    for square1 in chess.SQUARES:
        for square2 in chess.SQUARES:
            file_diff = chess.square_file(square2) - chess.square_file(square1)
            rank_diff = chess.square_rank(square2) - chess.square_rank(square1)
            if not diffs.get((file_diff, rank_diff)) is None:
                filter_table[square1, square2] = diffs[(file_diff, rank_diff)]

                if rank_diff == 1 and file_diff in [1,0,-1]:
                    filter_table[square1, square2, 2:5] = 3*(1+file_diff) + np.arange(64,67)
    return filter_table


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


def clean_pgn_file(filenames, game_filters=[], output_filename="cleaned_data.pgn", print_frequency=25000):
    """
    Creates a pgn file from the games contained in the given pgn files, after having removed games which
    did not pass a given set of filters.

    :param filenames: An iterable of the names of pgn files
    :param game_filters: A list of functions, each one accepting a python-chess Game object, and returning
     a boolean value indicating weather or not the board should be removed. The filters are applied sequentially
     (the second filter wouldn't be checked if the first filter returns True)
    :param output_filename: The filename used for saving the cleaned pgn
    :param print_frequency: The number of games checked between printing the progress
    """
    def game_is_okay(game):
        for filter in game_filters:
            if filter(game):
                return False
        return True

    with open(output_filename, 'w') as output_file:
        game_counter = 0

        for index, filename in enumerate(filenames):
            if print_frequency:
                print("Starting file", str(index + 1))
                prev_time = time.time()

            for cur_game in game_iterator(filename):
                if print_frequency and game_counter % print_frequency == 0:
                    print("%d games have been processed.  Time since last print: %f"%(game_counter, time.time() - prev_time))
                    prev_time = time.time()

                if game_is_okay(cur_game):
                    output_file.write("%s\n\n"%str(cur_game))

                game_counter += 1


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
    move_filter_table = generate_move_filter_table()

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

        move_filter = move_filter_table[
            move_made.from_square,
            move_made.to_square,
            NO_PROMOTION_VALUE if move_made.promotion is None else move_made.promotion]

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

    unique_structs = np.unique(np.array(collected))

    set_up_move_arrays(unique_structs)

    print("%d unique boards produced."%len(unique_structs))

    np.save(output_filename, unique_structs)





if __name__ == "__main__":
    """
    The commented out code throughout the rest of this file is how the ANN training databases are created,
    as well as any other database used in Batch First. 
    """


    def add_path_fn_creator(path):
        return lambda x : [path + y for y in x]

    PGN_FILENAMES = ["lichess_db_standard_rated_2018-06.pgn",
                 "lichess_db_standard_rated_2018-05.pgn",
                 "lichess_db_standard_rated_2018-04.pgn",
                 "lichess_db_standard_rated_2018-03.pgn",
                 "lichess_db_standard_rated_2018-02.pgn",
                 "lichess_db_standard_rated_2018-01.pgn",
                 "lichess_db_standard_rated_2017-12.pgn",
                 "lichess_db_standard_rated_2017-11.pgn",
                 "lichess_db_standard_rated_2017-10.pgn",
                 "lichess_db_standard_rated_2017-09.pgn",
                 ][::-1]

    PICKLE_FILENAMES = list(map(lambda s : s[:-3] + "pkl", PGN_FILENAMES))

    PGN_FILENAMES = add_path_fn_creator("/srv/databases/lichess/")(PGN_FILENAMES)

    FILTERED_FILENAMES = list(map(lambda f: "%s_filtered.pgn" % f[:-4], PGN_FILENAMES))

    TWO_PASS_TFRECORDS_FILENAMES = [
        "two_pass_lichess_training.tfrecords",
        "two_pass_lichess_validation.tfrecords",
        "two_pass_lichess_testing.tfrecords"]

    TFRECORDS_OUTPUT_RATIOS = [.85, .1, .05]





    # pgn_not_used_for_ann_training = "/srv/databases/lichess/lichess_db_standard_rated_2018-07.pgn"
    # npy_output_filename = "/srv/databases/lichess/lichess_db_standard_rated_2018-07_first_100k_games"
    # save_all_boards_from_game_as_npy(pgn_not_used_for_ann_training, npy_output_filename)




    # file_index = 0
    # OUTPUT_FILTERED_FILENAME = PGN_FILENAMES[file_index][:-4] + "_filtered.pgn"
    # clean_pgn_file(
    #     [PGN_FILENAMES[file_index]],
    #     game_filters=[
    #         lambda g : g.headers['Termination'] != "Normal",
    #         lambda g : g.variations and "%eval" not in str(g.variations[0])],
    #     output_filename=OUTPUT_FILTERED_FILENAME,
    #     print_frequency=25000)






    #########################The commented out code below is used for the move scoring database###################
    # file_index = 0
    # OUTPUT_FILENAME = "/srv/databases/lichess_just_move_scoring/TO_CONFIRM_GENERALITY.pkl"
    # CUR_MOVE_PGN_FILENAME = FILTERED_FILENAMES[file_index]
    #
    # get_data_from_pgns(
    #     [CUR_MOVE_PGN_FILENAME],
    #     OUTPUT_FILENAME,
    #     create_move_scoring_board_from_game_fn(5),
    # )


    #########################The commented out code below is used for the board evaluation database###################
    # file_index = 8
    # CUR_EVAL_PGN_FILENAME = FILTERED_FILENAMES[file_index]
    # OUTPUT_FILENAME = "/srv/databases/lichess_combined_methods_eval_databases/lichess_db_standard_rated_2018-05.pkl"
    #
    # get_data_from_pgns(
    #     [CUR_EVAL_PGN_FILENAME],
    #     OUTPUT_FILENAME,
    #     create_board_eval_board_from_game_fn(),
    # )






    #########################The commented out code below is used for the board evaluation database#########################
    # eval_path_adder = add_path_fn_creator("/srv/databases/lichess_combined_methods_eval_databases/")
    #
    # EVAL_PICKLE_FILES = eval_path_adder(PICKLE_FILENAMES)
    # EVAL_PICKLE_FILES += list(map(lambda s: s[:-4] + "_second_pass.pkl", EVAL_PICKLE_FILES))
    #
    # EVAL_OUTPUT_TFRECORDS_FILES = eval_path_adder(TWO_PASS_TFRECORDS_FILENAMES)
    #
    # combine_pickles_and_create_tfrecords(
    #     EVAL_PICKLE_FILES,
    #     EVAL_OUTPUT_TFRECORDS_FILES,
    #     TFRECORDS_OUTPUT_RATIOS,
    #     serializer_creator())


    #########################The commented out code below is used for the move scoring database###################
    # policy_path_adder_1 = add_path_fn_creator("/srv/databases/lichess_just_move_scoring/first_iteration_pickle_files/")
    # policy_path_adder_2 = add_path_fn_creator("/srv/databases/lichess_just_move_scoring/second_iteration_pickle_files/")
    # MOVE_SCORING_PICKLE_FILES = policy_path_adder_1(PICKLE_FILENAMES) + policy_path_adder_2(PICKLE_FILENAMES)
    # POLICY_OUTPUT_TFRECORDS_FILES = add_path_fn_creator("/srv/databases/lichess_just_move_scoring/")(TWO_PASS_TFRECORDS_FILENAMES)
    #
    # combine_pickles_and_create_tfrecords(
    #     MOVE_SCORING_PICKLE_FILES,
    #     POLICY_OUTPUT_TFRECORDS_FILES,
    #     TFRECORDS_OUTPUT_RATIOS,
    #     serializer_creator(additional_move_features))
