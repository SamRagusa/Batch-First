'''
Created on Aug 12, 2017

@author: SamRagusa
'''

import chess
import chess.pgn
import math
import time
from my_board import MyBoard
from ann_creation_helper import line_counter



class BoardInfo:
    """
    A class to store information relevant to a single game board, such as moves chosen by
    players (and their occurrences).
    """

    def __init__(self, move):
        """
        Creates a BoardInfo object, and updates it's data with the occurrence of the given move.
        """
        self.moves = {}
        self.update(move)

    def update(self, move):
        """
        A method to update the data from a given move.  It increments the counter for that move,
        or if one does not currently exist, it creates it and sets it to one.
        """
        if self.moves.get(move) is None:
            self.moves[move] = 1
        else:
            self.moves[move] += 1



def create_database_from_pgn(filenames, to_collect_filters=[], post_collection_filters=[], data_writer=None,
                             switch_black_move_to_white=True, fen_elements_stored=[0, 2],
                             output_filenames=["board_config_database.csv"]):
    """
    A function used to generate customized chess databases from a set of pgn files.  It does so using the python-chess
    package for pgn file parsing, move generation, and error handling (if the error lies within the pgn file itself).


    @param filenames An array of filenames (paths) for the png files of chess games to read data from.
    @param to_collect_filters   An array of functions used to filter out pairs of board positions and moves,
    doing so during the parsing of the pgn files.  Each function in the array must accept two parameters,
    the first being a string representation of the board (e.g. a comma separated string of the fen elements
    requested to be stored by the fen_elements_stored parameter),
    and the second being the BoardInfo object associated with that board.
    @param post_collection_filters An array of functions just like to_collect_filters, but applied after pgn
    parsing is completed, as opposed to during.
    @param data_writer A function that takes in two parameters, the first of which is a dictionary mapping string
    representations of boards (as determined by other parameters of this function) to BoardInfo objects,
    and the second parameter accepts the output_filenames array, which is given as another parameter of this function.
    @param switch_black_move_to_white True if boards where it's blacks turn should be switched to a representation where it's whites turn.
    @param fen_elements_stored The elements to be stored (and used for duplicate detection) from the output of
    chess.Board().fen().spit(), which is of the format:
    r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4
    @param output_filenames An array of filenames (paths) to be passed to the data_writer as a parameter.


    NOTES:
    1) Errors are currently being handled internally within the python-chess package, and the logs are being stored
    in the Game objects error array.
    """

    the_board = MyBoard()


    def black_fen_to_white():
        """
        Reformats and returns the output from a given boards fen() method from one where it's blacks turn,
        to one where it's whites turn.  This involves rotating the board along the x axis and switching the
        color of the pieces
        """
        split_fen = the_board.fen().split()

        split_fen[0] = "/".join(list(reversed([row.swapcase() for row in split_fen[0].split("/")])))
        split_fen[1] = "w"
        split_fen[2] = "".join(map(lambda x: x if (x in split_fen[2].swapcase()) else '', ['K', 'Q', 'k', 'q', '-']))
        if split_fen[3] != '-':
            split_fen[3] = split_fen[3][0] + str(9 - int(split_fen[3][1]))

        return ",".join(split_fen)


    configs = {}
    for index, filename in enumerate(filenames):

        print("Starting file", str(index + 1), "with", line_counter(filename), "lines")

        pgn_file = open(filename)


        cur_game = chess.pgn.read_game(pgn_file)
        while not cur_game is None:
            for move in cur_game.main_line():

                should_save = True
                if to_collect_filters != []:
                    for filter in to_collect_filters:
                        if filter(the_board, move):
                            should_save = False
                            break

                if should_save:

                    if switch_black_move_to_white and not the_board.turn:
                        white_move_string = black_fen_to_white()
                    else:
                        white_move_string = ",".join(the_board.fen().split())

                    white_move_string = ",".join([info for index, info in enumerate(white_move_string.split(",")) if
                                                  index in fen_elements_stored])

                    if configs.get(white_move_string) is None:
                        configs[white_move_string] = BoardInfo(move)
                    else:
                        configs[white_move_string].update(move)

                the_board.push(move)

            cur_game = chess.pgn.read_game(pgn_file)
            the_board.reset()

        pgn_file.close()

    print("Applying post-collection filters to data.")
    to_delete = []
    if post_collection_filters != []:
        for cur_str, data in configs.items():
            for filter in post_collection_filters:
                if filter(cur_str, data):
                    to_delete.append(cur_str)
                    break

        print("Number of boards deleted by post-collection filters:", len(to_delete))
        for cur_str in to_delete:
            del configs[cur_str]

    print("Writing data to new file.")
    if data_writer is None:
        writer = open(output_filenames[0], 'w')
        for cur_str, data in configs.items():
            writer.write(cur_str + "," + str(data) + "\n")
        writer.close()
    else:
        data_writer(configs, output_filenames)





def during_search_n_man_filter_creator(n):
    """
    Creates and returns a filter function for use in the create_database_from_pgn function, to be used during data acquisition.
    The function filters out boards which have less than or equal to a given number of chess pieces (n).
    """
    def filter(board, move):
        if chess.popcount(board.occupied) <= n:
            return True
        return False

    return filter


def during_search_capture_move_filter(board, move):
    """
    A filter function for use in the create_database_from_pgn function, to be used during data acquisition.
    Returns True (filters out the given info) if the given move is considered a capture for the given board.
    """
    if board.is_capture(move):
        return True
    return False


def during_search_leq_n_moves_made_filter_creater(n):
    """
    Creates and returns a filter function for use with the create_database_from_pgn function, to be used during data acquisition.
    The function filters out boards where the number of half-moves made is less than or equal to a given number (n).
    """

    def filter(board, move):
        if len(board.stack) <= n:
            return True
        return False

    return filter


def n_man_filter_creator(n):
    """
    Creates and returns a filter function for use in the create_database_from_pgn function.
    Returns True (filters out data) if the number of pieces in the given board configuration is greater than n,
    and False if there are less than or equal to n.
    """
    def filter(str, data):
        counter = 0
        for chr in str:
            if chr in "KQRBNPkqrbnp":
                counter += 1
                if counter > n:
                    return False
        return True

    return filter






def data_writer_creator(file_ratios, for_deep_pink_loss, comparison_move_generator=None, line_to_fen_fn=None,
                        board_to_str_fn=None, no_deep_pink_loss_str_manipulations=[], print_frequency=10000):
    """
    Creates a data writer for use in the create_database_from_pgn function.  It can create a set of database files
    containing both files that can be used with Deep Pink's loss function (or one based on a similar idea),
    and files containing only the board representations discovered.


    @param file_ratios The ratios of the given boards to use when creating each database file.
    @param for_deep_pink_loss An array of boolean values indicating which of the database files being created should be
    formatted for use with Deep Pink's loss function (or a loss function with a similar idea).
    @param comparison_move_generator A generator function taking a MyBoard and BoardInfo object as parameters.
    The generator produces moves such that from the given board, making one of the moves would result in
    the random board in the (old, new, random) triplet.
    @param line_to_fen_fn A function to convert from the string representation used as keys in
    the dictionary given to the data writer, to the fen representation accepted by the python-chess package.
    @param board_to_str_fn A function to convert a MyBoard object to the string representation desired for output.
    @param no_deep_pink_loss_str_manipulations An array of functions taking a string representation of a board,
    and returning a manipulated version of the string given.  These manipulations are currently only used when
    producing datasets not compatible with Deep Pink's loss function.
    @param print_frequency The increment in which to print the number of boards processed and written to file.


    NOTES:
    1) The switch_from_blacks_move function (contained within the writer function), does not work for every string
    representation given to it (by the board_to_str_fn function).
    2) The resulting data writer function is not nearly as fast as it could be.  A major speed increase
    would (likely) come from the use of the Numba move generation implementation as opposed to
    the python-chess implementation, even more so if the writer generated moves (among other things) from
    Numpy arrays all at once, allowing for easy parallelism (and potential GIL release).  Though this is not a priority.
    """

    if line_to_fen_fn is None:
        line_to_fen_fn = lambda x: x

    if board_to_str_fn is None:
        board_to_str_fn = lambda x: x

    if comparison_move_generator is None:
        comparison_move_generator = lambda b, d: b.legal_moves

    def writer(dict, filenames):
        """
        The function to be returned by data_writer_creator.
        """

        def switch_from_blacks_move(the_str):
            """
            Converts a string representation of a board in which it's blacks turn to move (output from
            board_to_str_fn function), to a representation where it's white's turn to move.

            IMPORTANT NOTES:
            1) This will not work on all string representations of the board, but works on current output of
            board_to_str_fn.  This function should probably be defined outside of the data_writer_creator function,
            and instead be given as a parameter.
            2) I believe this will also convert from white to black's turn.
            """
            return ("".join(list(reversed([the_str[8 * j: 8 * (j + 1)] for j in range(8)])))).swapcase()


        writers = [open(file, 'w') for file in filenames]

        number_of_boards = len(dict)

        print("Number of board configurations:", number_of_boards)

        cur_entry_num = 0

        dict_iterator = iter(dict.items())

        for writer, ratio, should_get_deep_pink_loss in zip(writers, file_ratios, for_deep_pink_loss):
            for _ in range(int(math.floor(ratio * number_of_boards))):

                if cur_entry_num % print_frequency == 0:
                    print(cur_entry_num, "moves writen.")

                cur_str, cur_data = next(dict_iterator)

                if should_get_deep_pink_loss:
                    temp_board = MyBoard(line_to_fen_fn(cur_str))

                    #Get the maximum number of times a move was chosen from the current position
                    max_move_count = max(cur_data.moves.values())

                    #Get the set of moves chosen max_move_count number of times
                    most_chosen_moves = [move for move in cur_data.moves.keys() if cur_data.moves.get(move) == max_move_count]

                    cur_board_rep = board_to_str_fn(temp_board)
                    data_strs = [cur_board_rep for _ in range(len(most_chosen_moves))]

                    for index, move in enumerate(most_chosen_moves):
                        temp_board.push(move)
                        data_strs[index] += switch_from_blacks_move(board_to_str_fn(temp_board))
                        temp_board.pop()

                    for move in comparison_move_generator(temp_board, cur_data):

                        temp_board.push(move)

                        str_to_add = switch_from_blacks_move(board_to_str_fn(temp_board)) + "\n"

                        writer.write("".join(map(lambda x: x + str_to_add, data_strs)))
                        temp_board.pop()
                else:
                    for manipulation in no_deep_pink_loss_str_manipulations:
                        cur_str = manipulation(cur_str)

                    #If the string representation used to create the Deep Pink datasets is preferred use this instead
                    #of the above for loop.
                    # cur_str = board_to_str_fn(MyBoard(line_to_fen_fn(cur_str)))

                    writer.write(cur_str + "\n")

                cur_entry_num += 1

            writer.close()

    return writer




def my_comparison_move_generator(board, data):
    """
    A move generator to produce moves such that from the given board, making one of the moves would result in
    the random board in the (old, new, random) triplet.  It generates legal moves for the given board which
    are not captures, and which were never picked by any player in the databases used.
    """
    for move in board.legal_moves:
        if not board.is_capture(move) and data.moves.get(move) is None:
            yield move



pgn_filenames = ["ficsgamesdb_1999_standard2000_nomovetimes_1505782.pgn",
                 "ficsgamesdb_2000_standard2000_nomovetimes_1505781.pgn",
                 "ficsgamesdb_2001_standard2000_nomovetimes_1505780.pgn",
                 "ficsgamesdb_2002_standard2000_nomovetimes_1505779.pgn",
                 "ficsgamesdb_2003_standard2000_nomovetimes_1505778.pgn",
                 "ficsgamesdb_2004_standard2000_nomovetimes_1505777.pgn",
                 "ficsgamesdb_2005_standard2000_nomovetimes_1490172.pgn",
                 "ficsgamesdb_2006_standard2000_nomovetimes_1490171.pgn",
                 "ficsgamesdb_2007_standard2000_nomovetimes_1490170.pgn",
                 "ficsgamesdb_2008_standard2000_nomovetimes_1490169.pgn",
                 "ficsgamesdb_2009_standard2000_nomovetimes_1490168.pgn",
                 "ficsgamesdb_2010_standard2000_nomovetimes_1490167.pgn",
                 "ficsgamesdb_2011_standard2000_nomovetimes_1490166.pgn",
                 "ficsgamesdb_2012_standard2000_nomovetimes_1490165.pgn",
                 "ficsgamesdb_2013_standard2000_nomovetimes_1490164.pgn",
                 "ficsgamesdb_2014_standard2000_nomovetimes_1490163.pgn",
                 "ficsgamesdb_2015_standard2000_nomovetimes_1490162.pgn",
                 "ficsgamesdb_2016_standard2000_nomovetimes_1490161.pgn"]


pgn_file_paths = list(map(lambda filename : "databases/fics/" + filename, pgn_filenames))


final_dataset_filenames = list(
    map(
        lambda x: "/srv/databases/chess_engine/pre_commit_test/" + x,
        [
            "scoring_training_set.txt",
            "scoring_validation_set.txt",
            "scoring_testing_set.txt",
            # "bns_training_set.csv",
            # "bns_validation_set.csv",
            # "tree_search_validation_set.csv",
            # "bns-tree_testing_set.csv"
            ]))


file_ratios = [.72, .08, .2]
for_deep_pink_loss_use = [True, True, True]



def current_line_to_fen(line):
    """
    A function to convert a board from a given string representation (the one used as keys in
    the dictionary given to the data writer), to the fen representation accepted by the python-chess package.

    In the current implementation, it is used to convert from a representation like the following:
    "2r3k1/pr3pp1/2n3b1/P1PR2Pp/4PB2/1N2QP1q/7P/1R4K1,-,-"
    """
    line.split(",")
    hacked_part_fen_array = line.split(",")
    hacked_part_fen_array.insert(1, "w")
    return " ".join(hacked_part_fen_array) + " 0 1"


#These manipulation are used to manipulate a string representation of a board.
manipulations = [(lambda y: (
    lambda x: ",".join([z if index != 0 else z.replace(str(y), "1" * y) for index, z in enumerate(x.split(","))])))(j) for j
                 in range(2, 9)]
manipulations.append(lambda x: x.replace("/", ""))








start_time = time.time()

create_database_from_pgn(
    pgn_file_paths[:-1],
    to_collect_filters=[
        during_search_n_man_filter_creator(6),
        during_search_leq_n_moves_made_filter_creater(5),
        during_search_capture_move_filter],
    fen_elements_stored=[0,2,3],
    data_writer=data_writer_creator(
        file_ratios,
        for_deep_pink_loss_use,
        comparison_move_generator=my_comparison_move_generator,
        line_to_fen_fn=current_line_to_fen,
        board_to_str_fn=lambda b : b.database_board_representation(),
        no_deep_pink_loss_str_manipulations=manipulations,   #This is not used in the current run configuration
        print_frequency=100000),
    output_filenames=final_dataset_filenames)

print("Time taken to create databases:", time.time() - start_time)




