import numba as nb
from numba import float32

from numba_board import *

from board_eval_client import ANNEvaluator
from global_open_priority_nodes import PriorityBins

import numpy as np

numpy_move_dtype = np.dtype([("from_square", np.uint8), ("to_square", np.uint8), ("promotion", np.uint8)])
move_type = nb.from_dtype(numpy_move_dtype)


# @jit(move_type[:,:](Move.class_type.instance_type[:]),nopython=True)
def create_move_record_array_from_move_object_generator(move_generator):
    return np.array([(move.from_square, move.to_square, move.promotion) for move in move_generator], dtype=numpy_move_dtype)


ANN_BOARD_EVALUATOR = ANNEvaluator(None, model_name="evaluation_ann")
ANN_MOVE_EVALUATOR_BLACK = ANNEvaluator(None, model_name="move_scoring_ann_white", model_desired_signature="legal_moves")      #THE ANNs DON'T SCORE BOARDS FOR THE RIGHT PLAYER SO THIS IS SWITCHED
ANN_MOVE_EVALUATOR_WHITE = ANNEvaluator(None, model_name="move_scoring_ann_black", model_desired_signature="legal_moves")

MIN_FLOAT32_VAL = np.finfo(np.float32).min
MAX_FLOAT32_VAL = np.finfo(np.float32).max
FLOAT32_EPS = np.finfo(np.float32).eps
ALMOST_MIN_FLOAT_32_VAL = MIN_FLOAT32_VAL + FLOAT32_EPS
ALMOST_MAX_FLOAT_32_VAL = MAX_FLOAT32_VAL - FLOAT32_EPS

WIN_RESULT_SCORE = ALMOST_MAX_FLOAT_32_VAL
LOSS_RESULT_SCORE = ALMOST_MIN_FLOAT_32_VAL
TIE_RESULT_SCORE = np.float32(0)


MOVE_TO_INDEX_ARRAY = np.zeros(shape=[64, 64], dtype=np.int32)
for key, value in generate_move_to_enumeration_dict().items():
    MOVE_TO_INDEX_ARRAY[key[0], key[1]] = value

gamenode_type = deferred_type()

gamenode_spec = OrderedDict()

gamenode_spec["board_state"] = BoardState.class_type.instance_type

# gamenode_spec["alpha"] = float32
# gamenode_spec["beta"] = float32
gamenode_spec["separator"] = float32
gamenode_spec["depth"] = uint8

gamenode_spec["parent"] = optional(gamenode_type)

gamenode_spec["best_value"] = float32

gamenode_spec["unexplored_moves"] = optional(move_type[:])

# No value in this array can be the minimum possible float value
gamenode_spec["unexplored_move_scores"] = optional(float32[:])

gamenode_spec["next_move_score"] = optional(float32)
gamenode_spec["next_move"] = optional(Move.class_type.instance_type)

gamenode_spec["terminated"] = boolean

# Using an 8 bit uint should work, since the maximum number of possible moves for a position seems to be 218.
gamenode_spec["children_left"] = uint8


@jitclass(gamenode_spec)
class GameNode:
    def __init__(self, board_state, parent, depth, separator, best_value, unexplored_moves, unexplored_move_scores, next_move_score, next_move, terminated, children_left):
        self.board_state = board_state

        # Eventually this will likely be some sort of list, so that updating multiple parents is possible
        # when handling transpositions.
        self.parent = parent

        # self.alpha = alpha
        # self.beta = beta
        self.separator = separator
        self.depth = depth

        self.best_value = best_value

        self.next_move_score = next_move_score
        self.next_move = next_move

        self.unexplored_moves = unexplored_moves
        self.unexplored_move_scores = unexplored_move_scores

        self.terminated = terminated

        self.children_left = children_left

    def zero_window_create_child_from_move(self, move):
        return GameNode(
            copy_push(self.board_state, move),
            self,
            self.depth - 1,
            -self.separator,
            MIN_FLOAT32_VAL,
            np.empty(0, dtype=numpy_move_dtype),
            None,
            None,
            None,
            False,
            np.uint8(0))

    def has_uncreated_child(self):
        if self.next_move_score is None:
            return False
        return True


gamenode_type.define(GameNode.class_type.instance_type)


@njit(GameNode.class_type.instance_type(uint8, float32))
def create_root_game_node(depth, seperator):
    return GameNode(create_initial_board_state(),
                    None,
                    depth,
                    seperator,
                    MIN_FLOAT32_VAL,
                    np.empty(0, dtype=numpy_move_dtype),
                    None,
                    None,
                    None,
                    False,
                    np.uint8(0))


def create_game_node_from_fen(fen, depth, seperator):
    return GameNode(create_board_state_from_fen(fen),
                    None,
                    depth,
                    seperator,
                    MIN_FLOAT32_VAL,
                    np.empty(0, dtype=numpy_move_dtype),
                    None,
                    None,
                    None,
                    False,
                    np.uint8(0))






# @jit(boolean(GameNode.class_type.instance_type),nopython=True)  #This worked, but prevented other functions from compiling
@njit
def should_terminate(game_node):
    cur_node = game_node
    while cur_node is not None:
        if cur_node.terminated:
            return True
        cur_node = cur_node.parent
    return False


# @jit((GameNode.class_type.instance_type, float32),nopython=True)
# @njit
def zero_window_update_node_from_value(node, value):
    if not node is None:
        if not node.terminated:  #This may be preventing more accurate predictions from zero-window search
            node.children_left -= 1
            node.best_value = max(value, node.best_value)

            if node.best_value >= node.separator or node.children_left == 0:
                node.terminated = True
                if not node.parent is None:  #This is being checked at the start of the call also and thus should be removed
                    temp_parent = node.parent
                    zero_window_update_node_from_value(temp_parent, -node.best_value)


# @jit(float32(BoardState.class_type.instance_type), nopython=True, nogil=True, cache=True)
# @njit #Can release the GIL here if wanted, but doesn't really speed things up
# def simple_board_state_evaluation_for_white(board_state):
#     score = np.float32(0)
#     for piece_val, piece_bb in zip(np.array([.1, .3, .45, .6, .9], dtype=np.float32), [board_state.pawns,board_state.knights , board_state.bishops, board_state.rooks, board_state.queens]):
#         score += piece_val * (np.float32(popcount(piece_bb & board_state.occupied_w)) - np.float32(popcount(piece_bb & board_state.occupied_b)))
#
#     return score


# @njit(float32(GameNode.class_type.instance_type))
# def evaluate_game_node(game_node):
#     """
#     A super simple evaluation function for a game_node.
#     """
#     return simple_board_state_evaluation_for_white(game_node.board_state)



@njit(void(GameNode.class_type.instance_type))
def set_up_next_best_move(node):
    next_move_index = np.argmax(node.unexplored_move_scores)

    if node.unexplored_move_scores[next_move_index] != MIN_FLOAT32_VAL:

        # Eventually the following line of code is what should be running instead of what is currently used
        # node.next_move = node.unexplored_moves[next_move_index]

        node.next_move = Move(node.unexplored_moves[next_move_index]["from_square"],
                              node.unexplored_moves[next_move_index]["to_square"],
                              node.unexplored_moves[next_move_index]["promotion"])

        node.next_move_score = node.unexplored_move_scores[next_move_index]
        node.unexplored_move_scores[next_move_index] = MIN_FLOAT32_VAL
    else:
        node.next_move = None
        node.next_move_score = None


@njit(GameNode.class_type.instance_type(GameNode.class_type.instance_type))
def spawn_child_from_node(node):
    to_return = node.zero_window_create_child_from_move(node.next_move)

    set_up_next_best_move(node)

    return to_return


def create_child_array(node_array):
    """
    Creates and returns an array of children spawned from an array of nodes who all have children left to
    be created, and who's move scoring has been initialized.
    """
    array_to_return = np.empty_like(node_array)
    for j in range(len(node_array)):
        array_to_return[j] = spawn_child_from_node(node_array[j])

    return array_to_return


def ann_evaluate_batch(node_array):
    """
    This is only used for the basic_do_alpha_beta function.
    """
    if len(node_array) != 0:
        return ANN_BOARD_EVALUATOR.for_numba_score_batch(node_array)
    return None


def async_ann_evaluate_node_batch(node_array):
    # Should be doing this much more efficiently
    if len(node_array) != 0:
        return ANN_BOARD_EVALUATOR.async_evaluate_boards_batch(node_array)


def async_score_moves(node_array):
    """
    Returns a tuple of a bool numpy array of the tuple of indices who's turn is white, and a size two array of futures
    for white and blacks move evaluation (or none if there were no nodes for white or black
    """
    if len(node_array) != 0:
        white_turn_indices = [True if node.board_state.turn else False for node in node_array]
        white_turn_nodes = node_array[white_turn_indices]
        black_turn_nodes = node_array[np.invert(white_turn_indices)]
        futures = [None, None]
        if len(white_turn_nodes) != 0:
            futures[0] = ANN_MOVE_EVALUATOR_WHITE.async_evaluate_move_batch(white_turn_nodes)
        if len(black_turn_nodes) != 0:
            futures[1] = ANN_MOVE_EVALUATOR_BLACK.async_evaluate_move_batch(black_turn_nodes)
        return white_turn_indices, futures
        # return None


# @profile
def get_indicies_of_terminating_children(node_array, hash_table):
    """
    This function checks for things such as the game being over for whatever reason,
    or information being found in the TT causing a node to terminate,
    or found in endgame tablebases,...

    Currently Doing:
    1) Checking if game is a checkmate or stalemate

    NOTES:
    1) In long term implementation will likely do a modification of starting a legal move generator
    and stopping when it reaches it's first move instead of doing it's entire move gen, but not yet...
    """
    # Creating zeros with dtype of np.bool will result in an array of False
    terminating = np.zeros(shape=len(node_array), dtype=np.bool)
    for j in range(len(node_array)):
        if False:  # not hash_table is None and terminated_from_tt(node_array[j], hash_table):
            terminating[j] = True
        else:
            legal_move_struct_array = create_move_record_array_from_move_object_generator(
                generate_legal_moves(node_array[j].board_state, BB_ALL, BB_ALL))

            if len(legal_move_struct_array) == 0:
                if is_in_check(node_array[j].board_state):
                    if node_array[j].board_state.turn == TURN_WHITE:
                        node_array[j].best_value = LOSS_RESULT_SCORE
                    else:
                        node_array[j].best_value = WIN_RESULT_SCORE
                else:
                    node_array[j].best_value = TIE_RESULT_SCORE

                # add_one_board_to_tt(node_array[j], hash_table)

                terminating[j] = True
            else:
                node_array[j].unexplored_moves = legal_move_struct_array
                node_array[j].children_left = np.uint8(len(legal_move_struct_array))

    return terminating


# @jit(void(GameNode.class_type.instance_type), nopython=True)
# def set_move_scores_to_random_values(node):
#     node.unexplored_move_scores = np.random.randn(len(node.unexplored_moves)).astype(np.float32)


# @jit(void(GameNode.class_type.instance_type), nopython=True)
# def set_move_scores_to_inverse_depth_multiplied_by_randoms(node):
#     scores = 1/(np.float32(node.depth)+1)*np.random.rand(len(node.unexplored_moves))
#     node.unexplored_move_scores = scores.astype(dtype=np.float32)


# def set_up_scored_move_generation(node_array):
#     for j in range(len(node_array)):
#         # set_move_scores_to_random_values(node_array[j])
#         set_move_scores_to_inverse_depth_multiplied_by_randoms(node_array[j])
#
#         set_up_next_best_move(node_array[j])


# @njit
# def ann_set_move_generation(node, scores, white_turn):
#     if white_turn:
#         move_indices = MOVE_TO_INDEX_ARRAY[node.unexplored_moves["from_square"], node.unexplored_moves["to_square"]]
#
#     else:
#         move_indices = MOVE_TO_INDEX_ARRAY[  ##########MAKE SURE UINTS DONT HAVE PROBLEMS HERE
#             8 * (7 - (node.unexplored_moves["from_square"] >> 3)) + (node.unexplored_moves["from_square"] & 7),
#             8 * (7 - (node.unexplored_moves["to_square"] >> 3)) + (node.unexplored_moves["to_square"] & 7)]
#
#     node.unexplored_move_scores = np.array([scores[index].score for index in move_indices], dtype=np.float32)


# @njit
# def jitted_ann_set_move_generation(node, scores, white_turn):
#     index_array = np.zeros(shape=len(node.unexplored_moves), dtype=np.int32)
#     if white_turn:
#         for j, move in enumerate(node.unexplored_moves):
#             index_array[j] = MOVE_TO_INDEX_ARRAY[node.unexplored_moves[j]['from_square'], node.unexplored_moves[j]["to_square"]]
#     else:
#         for j, move in enumerate(node.unexplored_moves):
#             index_array[j] = MOVE_TO_INDEX_ARRAY[
#                 8 * (7 - (node.unexplored_moves[j]["from_square"] >> 3)) + (node.unexplored_moves[j]["from_square"] & 7),
#                 8 * (7 - (node.unexplored_moves[j]["to_square"] >> 3)) + (node.unexplored_moves[j]["to_square"] & 7)]
#     node.unexplored_move_scores = scores[index_array]



# @profile
def ann_set_up_scored_move_generation(node_array, white_turn_indices, futures):
    white_move_nodes = node_array[white_turn_indices]
    black_move_nodes = node_array[np.invert(white_turn_indices)]

    if not futures[0] is None:
        white_results = futures[0].result().result.classifications
        for j in range(len(white_move_nodes)):
            white_move_nodes[j].unexplored_move_scores = np.array([move_score.score for move_score in white_results[j].classes], dtype=np.float32)

            set_up_next_best_move(white_move_nodes[j])

    if not futures[1] is None:
        black_results = futures[1].result().result.classifications
        for j in range(len(black_move_nodes)):
            black_move_nodes[j].unexplored_move_scores = np.array([-move_score.score for move_score in black_results[j].classes], dtype=np.float32)

            set_up_next_best_move(black_move_nodes[j])


def set_node_array_best_values_from_values(node_array, results, testing=False):
    for j, result in zip(range(len(node_array)), results):
        if testing:
            if node_array[j].parent is None:
                print("CHANGING BEST VALUE ON ROOT NODE FROM", node_array[j].best_value, "to", result.value if node_array[j].board_state.turn == TURN_WHITE else -result.value)

        if node_array[j].board_state.turn == TURN_WHITE:
            node_array[j].best_value = result.value
        else:
            node_array[j].best_value = - result.value


def update_tree_from_terminating_nodes(node_array):
    for j in range(len(node_array)):
        if not node_array[j].parent is None:
            zero_window_update_node_from_value(node_array[j].parent, - node_array[j].best_value)
        else:
            print("WE HAVE A ROOT NODE TERMINATING AND SHOULD PROBABLY DO SOMETHING ABOUT IT")


# @profile
def do_iteration(batch, hash_table, testing=False):
    depth_zero_indices = np.array(
        [True if batch[j].depth == 0 else False for j in range(len(batch))])
    depth_not_zero_indices = np.logical_not(depth_zero_indices)

    depth_zero_nodes = batch[depth_zero_indices]
    depth_not_zero_nodes = batch[depth_not_zero_indices]

    evaluation_result_future = async_ann_evaluate_node_batch(depth_zero_nodes)

    if len(depth_not_zero_nodes) != 0:
        children_created = create_child_array(depth_not_zero_nodes)

        # for every child which is not terminating set it's move list to the full set of legal moves possible
        terminating_children_indices = get_indicies_of_terminating_children(children_created, hash_table)

        not_terminating_children = children_created[np.logical_not(terminating_children_indices)]
        white_turn_indices, move_scoring_futures = async_score_moves(not_terminating_children)

        have_children_left_indices = np.array(
            [True if depth_not_zero_nodes[j].has_uncreated_child() else False for j in
             range(len(depth_not_zero_nodes))])

        depth_not_zero_with_more_kids_nodes = depth_not_zero_nodes[have_children_left_indices]

        if testing:
            for j in range(len(depth_not_zero_with_more_kids_nodes)):
                if depth_not_zero_with_more_kids_nodes[j].next_move is None or depth_not_zero_with_more_kids_nodes[
                    j].next_move_score is None:
                    print("A NODE WITH NO KIDS IS BEING TREATED AS IF IT HAD MORE CHILDREN!")

        if len(depth_zero_nodes) != 0:
            result_value = evaluation_result_future.result().result.regressions
            set_node_array_best_values_from_values(depth_zero_nodes, result_value, testing)

        NODES_TO_UPDATE_TREE_WITH = np.concatenate((depth_zero_nodes, children_created[terminating_children_indices]))
        update_tree_from_terminating_nodes(NODES_TO_UPDATE_TREE_WITH)


        ann_set_up_scored_move_generation(not_terminating_children, white_turn_indices, move_scoring_futures)
        # set_up_scored_move_generation(NODES_TO_SET_UP_SCORING_GEN_STUFF)

        return np.concatenate((depth_not_zero_with_more_kids_nodes, not_terminating_children))
    else:

        if len(depth_zero_nodes) != 0:
            result_value = evaluation_result_future.result().result.regressions
            set_node_array_best_values_from_values(depth_zero_nodes, result_value, testing)

        update_tree_from_terminating_nodes(depth_zero_nodes)
        return np.empty(0)



# @profile
def do_alpha_beta_search_with_bins(root_game_node, max_batch_size, bins_to_use, print_info=False, full_testing=False):
    open_node_holder = PriorityBins(bins_to_use, max_batch_size)

    hash_table = None
    next_batch = np.array([root_game_node], dtype=np.object)

    if print_info or full_testing:
        iteration_counter = 0
        total_nodes_computed = 0
        given_for_insert = 0
        start_time = time.time()

    while len(next_batch) != 0:
        if print_info or full_testing:
            num_open_nodes = len(open_node_holder)
            print("Starting iteration", iteration_counter, "with batch size of", len(next_batch), "and", num_open_nodes, "open nodes, max bin of", open_node_holder.largest_bin())
            total_nodes_computed += len(next_batch)

            if full_testing:
                if root_game_node.terminated or root_game_node.best_value > root_game_node.separator or root_game_node.children_left == 0:
                    print("ROOT NODE WAS TERMINATED OR SHOULD BE TERMINATED AT START OF ITERATION.")

        to_insert = do_iteration(next_batch, hash_table, full_testing)


        if print_info or full_testing:
            given_for_insert += len(to_insert)

            if full_testing:
                for j in range(len(to_insert)):
                    if to_insert[j] is None:
                        print("Found none in array being inserted into global nodes!")



        if print_info or full_testing:
            print("Iteration", iteration_counter, "completed producing", len(to_insert), "nodes to insert.")

        try_num_getting_batch = 1
        while True:
            # print(len(open_node_holder))
            next_batch = open_node_holder.insert_batch_and_get_next_batch(to_insert, max_batch_size * try_num_getting_batch)

            to_insert = np.array([])
            if len(next_batch) != 0 or (len(next_batch) == 0 and len(open_node_holder) == 0):
                break
            else:
                try_num_getting_batch = 1

        if print_info or full_testing:
            iteration_counter += 1

            if full_testing:
                for j in range(len(next_batch)):
                    if next_batch[j] is None:
                        print("Found None in next_batch!")
                    elif next_batch[j].terminated:
                        print("A terminated node was found in the newly created next_batch!")
                    elif should_terminate(next_batch[j]):
                        print("Found node which should terminate in the newly created next_batch!")
                    elif next_batch[j].children_left <= 0:
                        print("Found node with", next_batch[j].to_insert_cleaned,
                              "children left being in the newly created next_batch!")

    if print_info or full_testing:
        print("Number of iterations taken", iteration_counter, "in total time", time.time() - starting_time)
        print("Average time per iteration:", (time.time() - start_time) / iteration_counter)
        print("Total nodes computed:", total_nodes_computed)
        print("Total nodes given to linked list for insert", given_for_insert)

    return root_game_node.best_value


# @njit(float32(GameNode.class_type.instance_type, float32, float32, nb.int8))
def basic_do_alpha_beta(game_node, alpha, beta, color):
    """
    A simple alpha-beta algorithm to test my code.

    """
    has_move = False
    for _ in generate_legal_moves(game_node.board_state, BB_ALL, BB_ALL):
        has_move = True
        break

    if not has_move:
        if is_in_check(game_node.board_state):
            if color == 1:
                return MAX_FLOAT32_VAL
            return MIN_FLOAT32_VAL
        return 0

    if game_node.depth == 0:
        return color * ann_evaluate_batch(np.array([game_node]))[0].value

    best_value = MIN_FLOAT32_VAL
    for move in generate_legal_moves(game_node.board_state, BB_ALL, BB_ALL):
        v = - basic_do_alpha_beta(game_node.zero_window_create_child_from_move(move), -beta, -alpha, -color)
        best_value = max(best_value, v)
        alpha = max(alpha, v)
        if alpha >= beta:
            break

    return best_value


def ann_set_up_root_search_tree_node(fen, depth, seperator):
    root_node_as_array = np.array([create_game_node_from_fen(fen, depth, seperator)])
    get_indicies_of_terminating_children(root_node_as_array, None)
    white_indexes, futures = async_score_moves(root_node_as_array)
    ann_set_up_scored_move_generation(root_node_as_array, white_indexes, futures)
    return root_node_as_array[0]











print("Done compiling and starting run.\n")

# starting_time = time.time()
# perft_results = jitted_perft_test(create_initial_board_state(), 4)
# print("Depth 4 perft test results:", perft_results, "in time", time.time()-starting_time, "\n")

FEN_TO_TEST = "rn1qk2r/p1p1bppp/bp2pn2/3p4/2PP4/1P3NP1/P2BPPBP/RN1QK2R w KQkq - 4 8"  # "rn1qkb1r/p1pp1ppp/bp2pn2/8/2PP4/5NP1/PP2PP1P/RNBQKB1R w KQkq - 1 5"
DEPTH_OF_SEARCH = 4
MAX_BATCH_LIST = [500, 1000]  # [j**2 for j in range(40,0,-2)] + [5000]
SEPERATING_VALUE = -20.7
BINS_TO_USE = np.arange(15, -15, -.025)

starting_time = time.time()
for j in MAX_BATCH_LIST:
    root_node = ann_set_up_root_search_tree_node(FEN_TO_TEST, DEPTH_OF_SEARCH, SEPERATING_VALUE)
    print("Root node starting possible moves:", root_node.children_left)
    starting_time = time.time()
    # cur_value = do_alpha_beta_search(root_node, j, print_info=True)
    cur_value = do_alpha_beta_search_with_bins(root_node, j, BINS_TO_USE, print_info=False, full_testing=False)
    print("Root node children_left after search:", root_node.children_left)
    if cur_value == MIN_FLOAT32_VAL:
        print("MIN FLOAT VALUE")
    elif cur_value == LOSS_RESULT_SCORE:
        print("LOSS RESULT SCORE")
    print("Value:", cur_value, "found with max batch", j, "in time:", time.time() - starting_time)
    print()

print("Total white move sets evaluated:", ANN_MOVE_EVALUATOR_WHITE.total_evaluated_nodes, "with an average batch size of:", ANN_MOVE_EVALUATOR_WHITE.total_evaluated_nodes / ANN_MOVE_EVALUATOR_WHITE.total_evaluations)
print("Total black move sets evaluated:", ANN_MOVE_EVALUATOR_BLACK.total_evaluated_nodes, "with an average batch size of:", ANN_MOVE_EVALUATOR_BLACK.total_evaluated_nodes / ANN_MOVE_EVALUATOR_BLACK.total_evaluations)
print("Total boards scored:", ANN_BOARD_EVALUATOR.total_evaluated_nodes, "with an average batch size of:", ANN_BOARD_EVALUATOR.total_evaluated_nodes / ANN_BOARD_EVALUATOR.total_evaluations)

root_node = create_root_game_node(DEPTH_OF_SEARCH, -1)
for j in range(1):
    starting_time = time.time()
    print("Basic negamax with alpha-beta pruning resulted in:",
          basic_do_alpha_beta(root_node, MIN_FLOAT32_VAL, MAX_FLOAT32_VAL, 1), "in time:", time.time() - starting_time)
