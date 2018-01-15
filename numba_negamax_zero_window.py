import numpy as np
import numba as nb
from numba import int8, int32, uint8, uint16, uint64, float32, boolean, void, deferred_type, optional
from numba import jit, jitclass, vectorize
from numba import config
from collections import OrderedDict
from numba_board import *

import random as rand
from numpy import rec


numpy_move_dtype = np.dtype([("from_square", np.uint8), ("to_square", np.uint8), ("promotion", np.uint8)])
move_type = nb.from_dtype(numpy_move_dtype)


# @jit(move_type[:,:](Move.class_type.instance_type[:]),nopython=True)
def create_move_record_array_from_move_object_generator(move_generator):
    return np.array([(move.from_square, move.to_square, move.promotion) for move in move_generator], dtype=numpy_move_dtype)




MIN_FLOAT32_VAL = np.finfo(np.float32).min
MAX_FLOAT32_VAL = np.finfo(np.float32).max
FLOAT32_EPS = np.finfo(np.float32).eps
ALMOST_MIN_FLOAT_32_VAL = MIN_FLOAT32_VAL + FLOAT32_EPS
ALMOST_MAX_FLOAT_32_VAL = MAX_FLOAT32_VAL - FLOAT32_EPS


WIN_RESULT_SCORE = ALMOST_MAX_FLOAT_32_VAL
LOSS_RESULT_SCORE = ALMOST_MIN_FLOAT_32_VAL
TIE_RESULT_SCORE = np.float32(0)



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

#Using an 8 bit uint should work, since the maximum number of possible moves for a position seems to be 218.
gamenode_spec["children_left"] = uint8




@jitclass(gamenode_spec)
class GameNode:

    def __init__(self, board_state, parent, depth, separator, best_value,  unexplored_moves, unexplored_move_scores, next_move_score, next_move, terminated, children_left):
        self.board_state = board_state

        #Eventually this will likely be some sort of list, so that updating multiple parents is possible
        #when handling transpositions.
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



gamenode_type.define(GameNode.class_type.instance_type)





@jit(GameNode.class_type.instance_type(uint8, float32),  nopython=True)
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

@jit(boolean(GameNode.class_type.instance_type), nopython=True)
def has_uncreated_child(game_node):
    if game_node.next_move_score is None:
        return False
    return True


@jit(boolean(GameNode.class_type.instance_type),nopython=True)
def should_terminate(game_node):
    cur_node = game_node
    while cur_node is not None:
        if cur_node.terminated:
            return True
        cur_node = cur_node.parent
    return False


#@jit((GameNode.class_type.instance_type, float32),nopython=True)
def zero_window_update_node_from_value(node, value):
    if not node is None:
        if not node.terminated:
            #This complicated decrement is/was an attempt to have this function compile
            node.children_left = np.uint8(np.add(np.int8(node.children_left), np.int8(-1)))
            node.best_value = max(value, node.best_value)

            if node.best_value > node.separator or node.children_left == 0:
                node.terminated = True
                if not node.parent is None:
                    temp_parent = node.parent
                    zero_window_update_node_from_value(temp_parent, -node.best_value)


@jit(float32(BoardState.class_type.instance_type), nopython=True)#, nogil=True)
def simple_board_state_evaluation_for_white(board_state):
    score = np.float32(0)

    for piece_val, piece_bb in zip(np.array([.1, .3, .45, .6, .9], dtype=np.float32), [board_state.pawns,board_state.knights , board_state.bishops, board_state.rooks, board_state.queens]):
        score += piece_val * (np.float32(popcount(piece_bb & board_state.occupied_w)) - np.float32(popcount(piece_bb & board_state.occupied_b)))

    return score


@jit(float32(GameNode.class_type.instance_type), nopython=True)
def evaluate_game_node(game_node):
    """
    A super simple evaluation function for a game_node.
    """
    return simple_board_state_evaluation_for_white(game_node.board_state)



# @jit(void(GameNode.class_type.instance_type, float32), nopython=True)
# def update_game_node(game_node, value):
#     """
#     IMPORTANT NOTES:
#     1) This no longer works since I've switched the negamax implementation to only do zero window searchs,
#     though could be updated to work fairly easily
#     """
#     cur_game_node = game_node
#     cur_value = value
#     while not cur_game_node is None:
#         cur_game_node.best_value = max(cur_game_node.best_value, cur_value)
#         cur_game_node.alpha = max(cur_game_node.alpha, cur_value)
#         if cur_game_node.alpha >= cur_game_node.beta:
#
#             cur_game_node.terminated = True
#
#             cur_value = cur_game_node.best_value
#             cur_game_node = cur_game_node.parent
#
#
# @jit(void(GameNode.class_type.instance_type, int8), nopython=True)
# def do_alpha_beta(game_node, color):
#     """
#     A simple alpha-beta algorithm to test my code.
#
#     IMPORTANT NOTES:
#     1) This no longer works since I've switched the negamax implementation to only do zero window searchs,
#     though could be updated to work fairly easily
#     """
#     if game_node.depth == 0:
#         update_game_node(game_node, color * evaluate_game_node(game_node))
#
#     for move in generate_legal_moves(game_node.board_state, BB_ALL, BB_ALL):
#         update_game_node(game_node, - do_alpha_beta(create_child_from_move(game_node, move), -color))
#
#         if game_node.terminated:
#             break
#
    # return game_node.best_value





linked_list_node_type = deferred_type()

linked_list_node_spec = OrderedDict()
linked_list_node_spec["next"] = optional(linked_list_node_type)
linked_list_node_spec["data"] = gamenode_type


@jitclass(linked_list_node_spec)
class LinkedListNode:
    def __init__(self, next, data):
        self.next = next
        self.data = data



linked_list_node_type.define(LinkedListNode.class_type.instance_type)

priority_linked_list_spec = OrderedDict()
priority_linked_list_spec["root_node"] = optional(linked_list_node_type)
priority_linked_list_spec["max_batch_size"] = uint16
priority_linked_list_spec["size"] = uint64

@jitclass(priority_linked_list_spec)
class PriorityLinkedList:
    def __init__(self, root_node, max_batch_size, size):
        self.root_node = root_node
        self.max_batch_size = max_batch_size
        self.size = size



def print_linked_list_values(linked_list):
    cur_node = linked_list.root_node
    value_array = []
    while not cur_node is None:
        value_array.append(cur_node.data.next_move_score)
        cur_node = cur_node.next
    print("linked list size:",linked_list.size)
    print(value_array)


def check_linked_list_integrity(linked_list):
    """
    THINGS IT CHECKS:
    1) The size is correct
    2) The ordering is correct
    3) Every node in the linked list has been initialized in some way
    """
    cur_node = linked_list.root_node
    counter = 0
    min_score_found = MAX_FLOAT32_VAL
    times_out_of_order = 0
    times_move_has_no_next_move_score = 0
    while not cur_node is None:
        if cur_node.data.next_move_score is None or cur_node.data.next_move is None:
            times_move_has_no_next_move_score += 1
        elif cur_node.data.next_move_score > min_score_found:
            times_out_of_order += 1
            min_score_found = cur_node.data.next_move_score
        else:
            min_score_found = cur_node.data.next_move_score

        counter += 1
        cur_node = cur_node.next

    found_error = False

    if linked_list.size != counter:
        print("PriorityLinkedList.size is not correct, it's stored size is", linked_list.size,
              "but the actual size is", counter)
        found_error = True

    if times_out_of_order != 0:
        print("The linked list was found to be out of order", times_out_of_order, "times")
        found_error = True

    if times_move_has_no_next_move_score != 0:
        print("The linked list was found to have", times_move_has_no_next_move_score,
              "nodes without either a next move or a next move score.")
        found_error = True

    if found_error:
        print_linked_list_values(linked_list)


# @njit
def append_sorted_array_after_given_node(linked_list, sorted_array, final_node=None):
    """
    @param final_node The node in the linked list in which to start appending nodes after.  This parameter is None
    if the given linked list is empty.
    """
    cur_node = None

    for j in range(len(sorted_array) - 1, -1, -1):
        cur_node = LinkedListNode(cur_node, sorted_array[j])

    if final_node is None:
        linked_list.root_node = cur_node
    else:
        final_node.next = cur_node


#jit(void(PriorityLinkedList, GameNode.class_type.instance_type[:]), nopython=True)
#@njit
def push_sorted_array(linked_list, sorted_array):
    if linked_list.size == 0:
        append_sorted_array_after_given_node(linked_list, sorted_array)
    else:
        starting_node = LinkedListNode(linked_list.root_node, create_root_game_node(0,0))
        cur_node = starting_node
        cur_array_index = 0
        while cur_array_index != len(sorted_array):
            if sorted_array[cur_array_index].next_move_score > cur_node.next.data.next_move_score:
                cur_node.next = LinkedListNode(cur_node.next, sorted_array[cur_array_index])
                cur_array_index += 1
            else:
                if cur_node.next.next is None:
                    append_sorted_array_after_given_node(linked_list, sorted_array[cur_array_index:], cur_node.next)
                    break
            cur_node = cur_node.next

        linked_list.root_node = starting_node.next
    linked_list.size += len(sorted_array)


# @njit
def pop_full_list_as_numpy_array(linked_list):
    array_to_return = np.empty(shape=linked_list.size, dtype=np.object)
    index_in_return_array = 0
    while not linked_list.root_node is None:
        if not should_terminate(linked_list.root_node.data):
            array_to_return[index_in_return_array] = linked_list.root_node.data
            index_in_return_array += 1
        linked_list.root_node = linked_list.root_node.next

    linked_list.size = 0

    return array_to_return[:index_in_return_array]


def remove_consecutive_terminated_nodes_from_root(linked_list):
    while linked_list.size > 0 and should_terminate(linked_list.root_node.data):
        linked_list.root_node = linked_list.root_node.next
        linked_list.size -= 1


def create_next_batch_and_insert_sorted(linked_list, game_board_array, max_batch_size):
    """
    Takes in a priority linked list and sorted array of boards, and returns a batch (array) of boards within a given
    maximum size.  It does insertion and popping of nodes simultaneously to avoid going through the linked list twice.

    FUTURE IMPROVEMENTS:
    1) A likely dramatic speed improvement would be gained if instead of inserting the entire array of boards
    every time, the insertion stoped when the new batch has been created, and what remains of the sorted list is
    is stored with it's location in the linked list.  If during a future batch creation the location in the
    linked list is reached, it will continue inserting the list along with the current given list
    (and any other lists found this way).
    """

    remove_consecutive_terminated_nodes_from_root(linked_list)

    not_terminating_indices = np.array([False if should_terminate(game_board_array[j]) else True for j in range(len(game_board_array))], dtype=np.bool)

    game_board_array = game_board_array[not_terminating_indices]

    if linked_list.size + len(game_board_array) <= max_batch_size:
        #Just return the entire game_board_array and entire contents of the array list
        return np.concatenate((pop_full_list_as_numpy_array(linked_list), game_board_array))
    else:
        num_found = 0 #Pretty sure I don't need this anymore (and can use cur_return_array_index instead)
        cur_array_index = 0
        cur_return_array_index = 0
        # np.object array type won't work long term for JIT compilation (pretty sure)
        array_to_return = np.empty(shape=max_batch_size, dtype=np.object)

        while num_found < max_batch_size:
            if cur_array_index != len(game_board_array) and (linked_list.root_node is None or game_board_array[cur_array_index].next_move_score >= linked_list.root_node.data.next_move_score):
                array_to_return[cur_return_array_index] = game_board_array[cur_array_index]
                cur_return_array_index += 1
                cur_array_index += 1
            else:
                array_to_return[cur_return_array_index] = linked_list.root_node.data
                cur_return_array_index += 1
                linked_list.size -= 1
                linked_list.root_node = linked_list.root_node.next

                remove_consecutive_terminated_nodes_from_root(linked_list)

                #If the linked list has been exhausted
                if linked_list.size == 0:
                    if cur_array_index == len(game_board_array):
                        return array_to_return[:cur_return_array_index]
                    else:
                        if len(game_board_array) - cur_array_index > max_batch_size - cur_return_array_index:
                            array_to_return[cur_return_array_index:] = game_board_array[cur_array_index:cur_array_index + max_batch_size - cur_return_array_index]
                            push_sorted_array(linked_list, game_board_array[cur_array_index + max_batch_size - cur_return_array_index:])
                            return array_to_return
                        else:
                            array_to_return[cur_return_array_index:len(game_board_array) - cur_array_index + cur_return_array_index] = game_board_array[cur_array_index:]
                            return array_to_return[:len(game_board_array) - cur_array_index + cur_return_array_index]

            num_found += 1

        if cur_array_index != len(game_board_array):
            push_sorted_array(linked_list, game_board_array[cur_array_index:])

        return array_to_return


@jit(void(GameNode.class_type.instance_type), nopython=True)
def set_up_next_best_move(node):
    next_move_index = np.argmax(node.unexplored_move_scores)

    if node.unexplored_move_scores[next_move_index] != MIN_FLOAT32_VAL:

        #Eventually this is what should be happening
        # node.next_move = node.unexplored_moves[next_move_index]

        node.next_move = Move(node.unexplored_moves[next_move_index]["from_square"],
                              node.unexplored_moves[next_move_index]["to_square"],
                              node.unexplored_moves[next_move_index]["promotion"])

        node.next_move_score = node.unexplored_move_scores[next_move_index]
        node.unexplored_move_scores[next_move_index] = MIN_FLOAT32_VAL
    else:
        node.next_move = None
        node.next_move_score = None


# @jit(void(GameNode.class_type.instance_type), nopython=True)
def replace_next_best_move(node):
    if len(node.unexplored_moves) == 0:
        node.next_move = None
        node.next_move_score = None #probably don't need to do this
    else:
        set_up_next_best_move(node)


@jit(GameNode.class_type.instance_type(GameNode.class_type.instance_type),nopython=True)
def spawn_child_from_node(node):
    to_return = node.zero_window_create_child_from_move(node.next_move)

    # replace_next_best_move(node)

    set_up_next_best_move(node)

    return to_return


# @njit
def create_child_array(node_array):
    """
    Creates and returns an array of children spawned from an array of nodes who all have children left to
    be created, and who's move scoring has been initialized.
    """
    array_to_return = np.empty_like(node_array)
    for j in range(len(node_array)):
        array_to_return[j] = spawn_child_from_node(node_array[j])

    return array_to_return


# @njit
# @jit(void(GameNode.class_type.instance_type[:]), nopython=True)
def evaluate_depth_zero_batch(node_array):
    """
    Evaluates the nodes given, storing the value for any node N, in N.best_value.
    """
    for j in range(len(node_array)):
        if node_array[j].board_state.turn == TURN_WHITE: #This assumes that the player doing the search is white
            node_array[j].best_value = evaluate_game_node(node_array[j])
        else:
            node_array[j].best_value = -evaluate_game_node(node_array[j])


def get_indicies_of_terminating_children(node_array):
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
    #Creating zeros with dtype of np.bool will result in an array of False
    terminating = np.zeros(shape=len(node_array), dtype=np.bool)
    for j in range(len(node_array)):

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

            terminating[j] = True
        else:
            node_array[j].unexplored_moves = legal_move_struct_array
            node_array[j].children_left = np.uint8(len(legal_move_struct_array))

    return terminating


# @jit(void(GameNode.class_type.instance_type), nopython=True)
def set_move_scores_to_random_values(node):
    node.unexplored_move_scores = np.random.randn(len(node.unexplored_moves)).astype(np.float32)


# @jit(void(GameNode.class_type.instance_type), nopython=True)
def set_move_scores_to_inverse_depth_multiplied_by_randoms(node):
    scores = 1/(np.float32(node.depth)+1)*np.random.rand(len(node.unexplored_moves))
    node.unexplored_move_scores = scores.astype(dtype=np.float32)


# @jit(void(GameNode.class_type.instance_type), nopython=True)
def set_move_scores_to_all_zeros(node):
    """
    Set all move scores to 0 for a given node.
    """
    node.unexplored_move_scores = np.zeros(shape=len(node.unexplored_moves), dtype=np.float32)


def set_up_scored_move_generation(node_array):
    for j in range(len(node_array)):
        # set_move_scores_to_random_values(node_array[j])
        set_move_scores_to_inverse_depth_multiplied_by_randoms(node_array[j])

        set_up_next_best_move(node_array[j])


def update_tree_from_terminating_nodes(node_array):
    for j in range(len(node_array)):
        if not node_array[j].parent is None:

            zero_window_update_node_from_value(node_array[j].parent, - node_array[j].best_value)
        else:
            print("WE HAVE A ROOT NODE TERMINATING AND SHOULD PROBABLY DO SOMETHING ABOUT IT")


# @profile
def do_iteration(batch):
    depth_zero_indices = np.array(
        [True if batch[j].depth == 0 else False for j in range(len(batch))])
    depth_not_zero_indices = np.logical_not(depth_zero_indices)

    depth_zero_nodes = batch[depth_zero_indices]
    depth_not_zero_nodes = batch[depth_not_zero_indices]

    evaluate_depth_zero_batch(depth_zero_nodes)

    if len(depth_not_zero_nodes) != 0:
        children_created = create_child_array(depth_not_zero_nodes)

        #for every child which is not terminating set it's move list to the full set of legal moves possible
        terminating_children_indices = get_indicies_of_terminating_children(children_created)

        have_children_left_indices = np.array(
            [True if has_uncreated_child(depth_not_zero_nodes[j]) else False for j in range(len(depth_not_zero_nodes))])

        depth_not_zero_with_more_kids_nodes = depth_not_zero_nodes[have_children_left_indices]

        #for testing
        for j in range(len(depth_not_zero_with_more_kids_nodes)):
            if depth_not_zero_with_more_kids_nodes[j].next_move is None or depth_not_zero_with_more_kids_nodes[j].next_move_score is None:
                print("A NODE WITH NO KIDS IS BEING TREATED AS IF IT HAD MORE CHILDREN!")

        NODES_TO_UPDATE_TREE_WITH = np.concatenate((depth_zero_nodes, children_created[terminating_children_indices]))
        update_tree_from_terminating_nodes(NODES_TO_UPDATE_TREE_WITH)

        NODES_TO_SET_UP_SCORING_GEN_STUFF = children_created[np.logical_not(terminating_children_indices)]
        set_up_scored_move_generation(NODES_TO_SET_UP_SCORING_GEN_STUFF)

        CHILDREN_THAT_ARE_NOT_TERMINAL = children_created[np.logical_not(terminating_children_indices)]
        return np.concatenate((depth_not_zero_with_more_kids_nodes, CHILDREN_THAT_ARE_NOT_TERMINAL))
    else:
        update_tree_from_terminating_nodes(depth_zero_nodes)
        return np.empty(0)


# @profile
def do_alpha_beta_search(root_game_node, max_batch_size, testing=False):
    priority_node_queue = PriorityLinkedList(None, max_batch_size, 0)

    next_batch = np.array([root_game_node],dtype=np.object)

    if testing:
        iteration_counter = 0
        total_nodes_computed = 0
        given_for_insert = 0
        start_time = time.time()

    while len(next_batch) != 0:
        if testing:
            print("Starting iteration", iteration_counter, "with batch of size", len(next_batch), "and", priority_node_queue.size, "nodes on the queue.")
            total_nodes_computed += len(next_batch)

            if root_game_node.terminated or root_game_node.best_value > root_game_node.separator:
                print("ROOT NODE WAS TERMINATED OR SHOULD BE TERMINATED AT START OF ITERATION.")

        to_insert = do_iteration(next_batch)

        if testing:
            for j in range(len(to_insert)):
                if to_insert[j] is None:
                    print("Found None in array given to insert!")

            given_for_insert += len(to_insert)


        #sort the array of nodes to insert into the linked list
        values_to_sort = np.array([node.next_move_score for node in to_insert], dtype=np.float32)
        to_insert = to_insert[np.argsort(values_to_sort)]
        to_insert = to_insert[::-1]

        if testing:

            check_linked_list_integrity(priority_node_queue)
            print("Iteration", iteration_counter, "completed producing", len(to_insert), "nodes to insert.")


        next_batch = create_next_batch_and_insert_sorted(priority_node_queue, to_insert, max_batch_size)

        if testing:
            for j in range(len(next_batch)):
                if next_batch[j] is None:
                    print("Found None in next_batch!")
                elif next_batch[j].terminated:
                    print("A terminated node was found in the newly created next_batch!")

            check_linked_list_integrity(priority_node_queue)


            iteration_counter += 1

    if testing:
        print("Number of iterations taken", iteration_counter, "in total time", time.time() - starting_time)
        print("Average time per iteration:", (time.time() - start_time)/iteration_counter)
        print("Total nodes computed:", total_nodes_computed)
        print("Total nodes given to linked list for insert", given_for_insert)

    return root_game_node.best_value






print("Done compiling and starting run.\n")


starting_time = time.time()
perft_results = jitted_perft_test(create_initial_board_state(), 5)
print("Depth 4 perft test results:", perft_results, "in time", time.time()-starting_time, "\n")


DEPTH_OF_SEARCH = 4
MAX_BATCH_LIST = [j**2 for j in range(20,0,-1)]
for j in MAX_BATCH_LIST:
    root_node = create_root_game_node(DEPTH_OF_SEARCH,.1)

    get_indicies_of_terminating_children(np.array([root_node]))

    # set_move_scores_to_all_zeros(root_node)
    set_move_scores_to_random_values(root_node)

    set_up_next_best_move(root_node)
    starting_time = time.time()
    cur_value = do_alpha_beta_search(root_node, j)#, True)

    print("Value:", cur_value, "found with max batch", j, "in time:", time.time() - starting_time)

