from . import *

from .transposition_table import get_empty_hash_table, clear_hash_table
from .numba_negamax_zero_window import iterative_deepening_mtd_f, start_move_scoring, start_board_evaluations
from .global_open_priority_nodes import PriorityBins



def generate_bin_ranges(filename, move_eval_fn, percentiles=None, max_batch_size=1000, output_filename=None, print_info=False):
    """
    Generate values representing the boundaries for the bins in the PriorityBins class based on a given
    move evaluation function.  It also calculates the mean (zero-shift) for the move values.

    :param filename: The filename for the binary file (in NumPy .npy format) containing board structs.
     It's used for computing a sample of move scores
    :param move_eval_fn: The move evaluation function to be used when searching the tree
    :param percentiles: The percentiles desired from the sample of move scores computed
    :param max_batch_size: The maximum batch size to be given to the move_eval_fn
    :param output_filename: The filename to save the computed bins to, or None if saving the bins is not desired
    :param print_info: A boolean value indicating if info about the computations should be printed
    :return: An ndarray of the values at the given percentiles
    """
    def bin_helper_move_scoring_fn(struct_array):
        move_thread, move_score_getter, _, _ = start_move_scoring(
            struct_array,
            struct_array[:1],
            np.ones(len(struct_array), dtype=np.bool_),
            np.zeros(1, dtype=np.bool_),
            move_eval_fn)

        to_concat_from_squares = []
        to_concat_filters = []

        for struct in struct_array:
            relevant_moves = struct['unexplored_moves'][:struct['children_left']]

            if not struct['turn']:
                relevant_moves[:,:2] = SQUARES_180[relevant_moves[:,:2]]

            to_concat_filters.append(MOVE_FILTER_LOOKUP[relevant_moves[:, 0], relevant_moves[:, 1], relevant_moves[:, 2]])
            to_concat_from_squares.append(relevant_moves[:,0])

        move_filters = np.concatenate(to_concat_filters)
        from_squares = np.concatenate(to_concat_from_squares)


        move_thread.join()

        to_return = move_score_getter[0]([move_filters, from_squares, struct_array['children_left']])

        return to_return


    if percentiles is None:
        percentiles = np.arange(0, 100, .02)

    if print_info:
        print("Loading data from file for bin calculations")

    struct_array = np.load(filename)

    if print_info:
        print("Loaded %d BoardInfo structs"%len(struct_array))


    increment = max_batch_size
    struct_array = struct_array[:- (len(struct_array) % increment)]

    combined_results = np.concatenate(
        [bin_helper_move_scoring_fn(struct_array[j * increment:(j + 1) * increment]) for j in range((len(struct_array)-1)//increment)])

    zero_shift = np.mean(combined_results)

    combined_results -= zero_shift
    combined_results = np.abs(combined_results)

    if print_info:
        print("Computed %d move evaluations"%len(combined_results))

    bins = np.percentile(combined_results, percentiles)

    if not output_filename is None:
        np.save(output_filename, bins)
        np.save(output_filename + "_shift", zero_shift)

        if print_info:
            print("Saved bins to file")

    return bins, zero_shift


def calculate_eval_zero_shift(filename, board_eval_fn, max_batch_size=5000, output_filename="draw_board_mean", print_info=False):
    """
    Calculates the mean evaluation value of boards which have the 'expected' value of 0 (currently decided by StockFish).
    The calculated mean can then be used to shift the evaluation function so that it values tie games
    at 0 (the new evaluation is calculated: f'(x)=f(x)-mean).

    Zero-shifting the evaluation function is crucial since negamax is used rather than minimax, and because
    it maintains the accuracy of the stored draw value.


    :param filename: The filename for the binary file (in NumPy .npy format) containing board structs which each
    have a 'desired' value of 0 (according to StockFish).
     It's used for computing a sample of move scores
    :param board_eval_fn: The board evaluation function to be used when searching the tree
    :param max_batch_size: The maximum batch size to be given to the board_eval_fn
    :param output_filename: The filename to save the computed bins to, or None if saving the bins is not desired
    :param print_info: A boolean value indicating if info about the computations should be printed
    :return: The mean evaluation value
    """
    def eval_helper(struct_array):
        thread, scores = start_board_evaluations(
            struct_array,
            np.ones(len(struct_array), dtype=np.bool_),
            board_eval_fn)

        thread.join()

        return scores

    if print_info:
        print("Loading data from file for zero-shift calculations")

    struct_array = np.load(filename)

    if print_info:
        print("Loaded %d BoardInfo structs"%len(struct_array))

    struct_array = struct_array[:- (len(struct_array) % max_batch_size)]

    num_batches = (len(struct_array)-1)//max_batch_size

    combined_results = np.concatenate(
        [eval_helper(
            struct_array[j * max_batch_size:(j + 1) * max_batch_size]) for j in range(num_batches)])

    if print_info:
        print("Computed %d board evaluations"%len(combined_results))

    mean = np.float32(np.mean(combined_results))

    if print_info:
        print("The mean calculated board evaluation value is: %f"%mean)

    if not output_filename is None:
        np.save(output_filename, mean)

        if print_info:
            print("Saved mean to file")

    return mean


def get_previous_board_map_from_py_board(board):
    """
    SPEED IMPROVEMENTS TO MAKE:
    1) Stop going backward if castling rights are changed
    2) Stop using the python-chess board implementation
    3) Stop using a dictionary so it can be JIT compiled
    """
    board = board.copy()  #Just in case the board shouldn't be modified

    hash_dict = {zobrist_hash(board):1}

    while board.move_stack and board.halfmove_clock:
        board.pop()

        cur_hash = np.int64(np.uint64(zobrist_hash(board)))
        if cur_hash in hash_dict:
            hash_dict[cur_hash] += 1
        else:
            hash_dict[cur_hash] = 1

    to_pass_on = np.array(list(zip(*hash_dict.items())), dtype=np.uint64)

    return to_pass_on[:, np.argsort(to_pass_on[0])]




class ChessEngine(object):

    def pick_move(self, Board):
        """
        Given a Python-Chess Board object, return a Python-Chess Move object representing the move
        the engine would like to make.
        """
        raise NotImplementedError("This method must be implemented!")

    def start_new_game(self):
        """
        Run at the start of each new game
        """
        pass

    def ready_engine(self):
        """
        Set up whatever is needed to choose a move (used if resources must be released after each move).
        """
        pass

    def release_resources(self):
        """
        Release the resources currently used by the engine (like GPU memory or large chunks of RAM).
        """
        pass




class BatchFirstEngine(ChessEngine):

    def __init__(self, search_depth, board_eval_fn, move_eval_fn, bin_database_file=None, bin_output_filename=None,
                 first_guess_fn=None, max_batch_size=5000, zero_valued_boards_file=None, saved_zero_shift_file=None):
        """
        :param bin_database_file: If bin_output_filename is not None, then this is the NumPy database of boards to have
        bins be created from.  If bin_output_filename is None, then this is the NumPy file containing an array of bins.
        :param bin_output_filename: The name of the (NumPy) file which will be saved containing the bins computed, or None
        if the bins should not be saved.
        """
        if saved_zero_shift_file is None and zero_valued_boards_file is None:
            raise ValueError("Either saved_zero_shift_file or zero_valued_boards_file must be specified, but both are None!")

        if bin_database_file is None and bin_output_filename is None:
            raise ValueError("Either bin_database_file or bin_output_filename must be specified but both are None!")


        if first_guess_fn is None:
            self.first_guess_fn = lambda x : 0
        else:
            self.first_guess_fn = first_guess_fn

        self.search_depth = search_depth

        self.board_evaluator = board_eval_fn
        self.move_evaluator = move_eval_fn

        if bin_output_filename is None:
            self.bins = np.load(bin_database_file)
            move_zero_shift = np.load(bin_database_file[:-4] + "_shift.npy")
        else:
            self.bins, move_zero_shift = generate_bin_ranges(
                bin_database_file,
                self.move_evaluator,
                max_batch_size=int(1.25*max_batch_size),
                output_filename=bin_output_filename,
                print_info=True)


        if zero_valued_boards_file is None:
            zero_shift = np.load(saved_zero_shift_file)
        else:
            zero_shift = calculate_eval_zero_shift(
                zero_valued_boards_file,
                self.board_evaluator,
                max_batch_size=int(1.25*max_batch_size),
                output_filename=saved_zero_shift_file,
                print_info=True)

        self.board_evaluator = lambda *args : board_eval_fn(*args) - zero_shift

        self.open_node_holder = PriorityBins(
            self.bins,
            max_batch_size,
            zero_shift=move_zero_shift,
            # save_info=True, #Must be set to True if printing info about the searches!
        )

        self.hash_table = get_empty_hash_table()

    def start_new_game(self):
        clear_hash_table(self.hash_table)

    def pick_move(self, board):
        returned_score, move_to_return, self.hash_table = iterative_deepening_mtd_f(
            fen=board.fen(),
            depths_to_search=np.arange(1,self.search_depth+1),
            open_node_holder=self.open_node_holder,
            board_eval_fn=self.board_evaluator,
            move_eval_fn=self.move_evaluator,
            hash_table=self.hash_table,

            previous_board_map=get_previous_board_map_from_py_board(board),

            # print_info=True,       #If this is True, the save_info parameter for the PriorityBins must be True (in the __init__ function)  (better connection of these values to come)!
            )

        return move_to_return

