import numpy as np
from scipy import stats

from .transposition_table import get_empty_hash_table
from .numba_negamax_zero_window import iterative_deepening_mtd_f, start_move_scoring
from .global_open_priority_nodes import PriorityBins



def generate_bin_ranges(filename, move_eval_fn, quantiles=None, print_info=False, num_batches=125):
    """
    Generate values representing the boundaries for the bins in the PriorityBins class based on a given
    move evaluation function.

    :param filename: The filename for the binary file (in NumPy .npy format) containing board structs.
     It's used for computing a sample of move scores
    :param move_eval_fn: The move evaluation function to be used when searching the tree
    :param quantiles: The quantiles desired from the sample of move scores computed
    :param print_info: A boolean value indicating if info about the computations should be printed
    :param num_batches: The number of batches to split the given database into for inference
    :return: An ndarray of the values at the given quantiles
    """
    def bin_helper_move_scoring_fn(struct_array, move_eval_fn):
        move_thread, move_score_getter, _, _ = start_move_scoring(
            struct_array,
            struct_array[:1],
            np.ones(len(struct_array), dtype=np.bool_),
            np.zeros(1, dtype=np.bool_),
            move_eval_fn)

        from_to_squares = np.concatenate([struct['unexplored_moves'][:struct['children_left'], :2] for struct in struct_array])

        move_thread.join()

        return - move_score_getter[0]([from_to_squares, struct_array['children_left']])


    if quantiles is None:
        quantiles = np.arange(0, 1, .001)

    if print_info:
        print("Loading data from file for bin calculations")

    struct_array = np.load(filename)

    if print_info:
        print("Loaded %d BoardInfo structs"%len(struct_array))

    increment = len(struct_array)//num_batches
    results = []

    for j in range(num_batches):
        results.append(
            bin_helper_move_scoring_fn(
                struct_array[j*increment:(j+1)*increment],
                move_eval_fn))

    combined_results = np.concatenate(results)

    if print_info:
        print("Computed %d move evaluations"%len(combined_results))

    return stats.mstats.mquantiles(combined_results, quantiles)


class ChessEngine(object):

    def pick_move(self, Board):
        """
        Given a Python-Chess Board object, return a Python-Chess Move object representing the move
        the engine would like to make.
        """
        raise NotImplementedError("This method must be implemented!")

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

    def __init__(self, search_depth, board_eval_fn, move_eval_fn, bin_database_file, win_threshold=100000, loss_threshold=-100000, first_guess_fn=None):
        if first_guess_fn is None:
            self.first_guess_fn = lambda x : 0
        else:
            self.first_guess_fn = first_guess_fn

        self.search_depth = search_depth
        self.hash_table = get_empty_hash_table()

        self.board_evaluator = board_eval_fn
        self.move_evaluator = move_eval_fn

        self.open_node_holder = PriorityBins(
            generate_bin_ranges(bin_database_file, self.move_evaluator),
            10000,
            testing=False)

        self.win_threshold = win_threshold
        self.loss_threshold = loss_threshold

    def pick_move(self, board):
        returned_score, move_to_return, self.hash_table = iterative_deepening_mtd_f(
            fen=board.fen(),
            depths_to_search=np.arange(self.search_depth)+1,
            min_windows_to_confirm=[.001]*self.search_depth,
            open_node_holder=self.open_node_holder,
            board_eval_fn=self.board_evaluator,
            move_eval_fn=self.move_evaluator,
            hash_table=self.hash_table,
            win_threshold=self.win_threshold,
            loss_threshold=self.loss_threshold,
            # print_partial_info=True,
            # print_all_info=True,         #If this is True, print_partial_info must also be True!
            # testing=True,
            )




        return move_to_return
