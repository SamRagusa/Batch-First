import numpy as np

from .transposition_table import get_empty_hash_table
from .numba_negamax_zero_window import iterative_deepening_mtd_f, start_move_scoring
from .global_open_priority_nodes import PriorityBins


def generate_bin_ranges(filename, move_eval_fn, percentiles=None, print_info=False, num_batches=1000, output_filename=None):
    """
    Generate values representing the boundaries for the bins in the PriorityBins class based on a given
    move evaluation function.

    :param filename: The filename for the binary file (in NumPy .npy format) containing board structs.
     It's used for computing a sample of move scores
    :param move_eval_fn: The move evaluation function to be used when searching the tree
    :param percentiles: The percentiles desired from the sample of move scores computed
    :param print_info: A boolean value indicating if info about the computations should be printed
    :param num_batches: The number of batches to split the given database into for inference
    :param output_filename: The filename to save the computed bins to, or None if saving the bins is not desired
    :return: An ndarray of the values at the given percentiles
    """
    def bin_helper_move_scoring_fn(struct_array):
        move_thread, move_score_getter, _, _ = start_move_scoring(
            struct_array,
            struct_array[:1],
            np.ones(len(struct_array), dtype=np.bool_),
            np.zeros(1, dtype=np.bool_),
            move_eval_fn)

        from_to_squares = np.concatenate([struct['unexplored_moves'][:struct['children_left'], :2] for struct in struct_array])

        move_thread.join()

        return - move_score_getter[0]([from_to_squares, struct_array['children_left']])


    if percentiles is None:
        percentiles = np.arange(0, 100, .1)

    if print_info:
        print("Loading data from file for bin calculations")

    struct_array = np.load(filename)

    if print_info:
        print("Loaded %d BoardInfo structs"%len(struct_array))

    increment = len(struct_array)//num_batches

    combined_results = np.concatenate(
        [bin_helper_move_scoring_fn(struct_array[j * increment:(j + 1) * increment]) for j in range(num_batches)])


    if print_info:
        print("Computed %d move evaluations"%len(combined_results))

    bins = np.percentile(combined_results, percentiles)

    if not output_filename is None:
        np.save(output_filename, bins)

        if print_info:
            print("Saved bins to file")

    return bins


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

    def __init__(self, search_depth, board_eval_fn, move_eval_fn, bin_database_file, bin_output_filename=None,
                 first_guess_fn=None, max_batch_size=5000):
        """
        :param bin_database_file: If bin_output_filename is not None, then this is the NumPy database of boards to have
        bins be created from.  If bin_output_filename is None, then this is the NumPy file containing an array of bins.
        :param bin_output_filename: The name of the (NumPy) file which will be saved containing the bins computed, or None
        if the bins should not be saved.
        """
        if first_guess_fn is None:
            self.first_guess_fn = lambda x : 0
        else:
            self.first_guess_fn = first_guess_fn

        self.search_depth = search_depth
        self.hash_table = get_empty_hash_table()

        self.board_evaluator = board_eval_fn
        self.move_evaluator = move_eval_fn

        if bin_output_filename is None:
            self.bins = np.load(bin_database_file)
        else:
            self.bins = generate_bin_ranges(
                bin_database_file,
                self.move_evaluator,
                output_filename=bin_output_filename,
                print_info=True)

        self.open_node_holder = PriorityBins(
            self.bins,
            max_batch_size,
            testing=False)

    def start_new_game(self):
        self.hash_table = get_empty_hash_table()

    def pick_move(self, board):
        returned_score, move_to_return, self.hash_table = iterative_deepening_mtd_f(
            fen=board.fen(),
            depths_to_search=np.arange(1,self.search_depth+1),
            open_node_holder=self.open_node_holder,
            board_eval_fn=self.board_evaluator,
            move_eval_fn=self.move_evaluator,
            hash_table=self.hash_table,
            # print_partial_info=True,
            # print_all_info=True,         #If this is True, print_partial_info must also be True!
            # testing=True,
            )

        return move_to_return

