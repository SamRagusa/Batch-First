import threading
from collections import OrderedDict
import chess.uci
import time

from .numba_board import *
from .global_open_priority_nodes import PriorityBins
from .anns.move_evaluation_ann import main as move_ann_main
from .board_jitclass import BoardState, generate_move_to_enumeration_dict, has_legal_move, is_in_check
from .transposition_table import get_empty_hash_table, add_board_and_move_to_tt
from .chestimator import get_inference_functions




# import llvmlite.binding as llvm
# llvm.set_option('', '--debug-only=loop-vectorize')





#(IMPORTANT) the constants below are only here to support older ANNs which use an the older mapping of moves
#(defined in board_jitclass). They use the same names as those in engine_constants.py,
# so if using the newer mapping, this must be prevented.
MOVE_TO_INDEX_ARRAY = np.zeros(shape=[64,64],dtype=np.int32)
REVERSED_MOVE_TO_INDEX_ARRAY = np.zeros(shape=[64,64],dtype=np.int32)
for key, value in generate_move_to_enumeration_dict().items():
    MOVE_TO_INDEX_ARRAY[key[0],key[1]] = value
    REVERSED_MOVE_TO_INDEX_ARRAY[square_mirror(key[0]), square_mirror(key[1])] = value












new_gamenode_type = nb.deferred_type()

new_gamenode_spec = OrderedDict()

#Ideally this would be a single Record, but I couldn't get that to compile yet
new_gamenode_spec["board_struct"] = numba_node_info_type[:]

new_gamenode_spec["parent"] = nb.optional(new_gamenode_type)


@nb.jitclass(new_gamenode_spec)
class GameNode:
    def __init__(self, board_struct, parent):
        self.board_struct = board_struct

        # Eventually this will be some sort of list, so that updating multiple parents is possible
        # when handling transpositions which are open at the same time.
        self.parent = parent


new_gamenode_type.define(GameNode.class_type.instance_type)







@njit(nogil=True)
def struct_array_to_ann_input_all_white(struct_array):
    ep_squares = np.zeros(len(struct_array),dtype=np.uint8)
    potential_reversal = np.empty((9,len(struct_array)), dtype=np.uint64)

    potential_reversal[0] = struct_array['kings']
    potential_reversal[1] = struct_array['queens']
    potential_reversal[2] = struct_array['rooks'] ^ struct_array['castling_rights']   #non castling rooks
    potential_reversal[3] = struct_array['bishops']
    potential_reversal[4] = struct_array['knights']
    potential_reversal[5] = struct_array['pawns']

    potential_reversal[6] = struct_array['castling_rights']

    for j in range(len(struct_array)):

        if struct_array[j]['turn']:
            if struct_array[j]['ep_square'] != 0:
                ep_squares[j] = struct_array[j]['ep_square']

            potential_reversal[7,j] = struct_array[j]['occupied_w']
            potential_reversal[8,j] = struct_array[j]['occupied_b']
        else:
            if struct_array[j]['ep_square'] != 0:
                ep_squares[j] = REVERSED_SQUARES[struct_array[j]['ep_square']]

            potential_reversal[7,j] = struct_array[j]['occupied_b']
            potential_reversal[8,j] = struct_array[j]['occupied_w']

    # to_return[10] = struct_array['occupied']

    black_turn_mask = np.logical_not(struct_array['turn'])
    potential_reversal[:, black_turn_mask] = vectorized_flip_vertically(potential_reversal[:,black_turn_mask])

    return np.expand_dims(potential_reversal[:7].transpose(), 1), np.expand_dims(potential_reversal[7:].transpose(), 2), ep_squares


@njit
def set_up_next_best_move(board_struct):
    board_struct['next_move_index'] = np.argmax(board_struct['unexplored_move_scores'])
    best_move_score = board_struct['unexplored_move_scores'][board_struct['next_move_index']]
    if best_move_score == MIN_FLOAT32_VAL:
        board_struct['next_move_index'] = 255
    else:
        board_struct['unexplored_move_scores'][board_struct['next_move_index']] = MIN_FLOAT32_VAL
    return best_move_score



@njit(nogil=True)
def jitted_assign_move_scores(children, not_children, children_indices, not_children_indices, size_array, results, child_best_move_scores, not_child_best_move_scores):
    num_children = len(children_indices)
    child_cum_sum_sizes = np.cumsum(size_array[:num_children])
    not_child_cum_sum_sizes = np.cumsum(size_array[num_children:])

    for j in range(len(children_indices)):
        children[children_indices[j]]['unexplored_move_scores'][:size_array[j]] = results[child_cum_sum_sizes[j] - size_array[j]:child_cum_sum_sizes[j]]
        child_best_move_scores[j] = set_up_next_best_move(children[children_indices[j]])

    for j in range(len(not_children_indices)):
        not_children[not_children_indices[j]]['unexplored_move_scores'][:size_array[j + num_children]] = results[not_child_cum_sum_sizes[j] -size_array[j + num_children]:not_child_cum_sum_sizes[j]]
        not_child_best_move_scores[j] = set_up_next_best_move(not_children[not_children_indices[j]])


@njit(nogil=True)
def move_scoring_helper(children, not_children, children_indices, not_children_indices):
    if len(children_indices) != 0:
        children_to_score = children[children_indices]

    if len(not_children_indices) != 0:
        not_children_to_score = not_children[not_children_indices]

    combined_to_score = np.concatenate((children_to_score, not_children_to_score))
    piece_bbs, occupied_bbs, ep_squares = struct_array_to_ann_input_all_white(combined_to_score)

    # This should REALLY be using views of combined_to_score['children_left'], since there will never be 0 children_left or more than 255 (meaning a uint8 should work to represent the data)
    size_array = combined_to_score[:]['children_left'].astype(np.int16)
    size_array[len(children_indices):][:] -= 1

    from_to_squares = np.empty((np.sum(size_array),2),dtype=np.int32)
    cur_start_index = 0
    for j in range(len(combined_to_score)):

        if combined_to_score[j]['turn']:
            from_to_squares[cur_start_index:cur_start_index + size_array[j], :] = combined_to_score[j]['unexplored_moves'][:size_array[j],:2]
        else:
            #These two lines should be done in 1 using slicing like above
            from_to_squares[cur_start_index:cur_start_index + size_array[j], 0] = REVERSED_SQUARES[combined_to_score[j]['unexplored_moves'][:size_array[j], 0]]
            from_to_squares[cur_start_index:cur_start_index + size_array[j], 1] = REVERSED_SQUARES[combined_to_score[j]['unexplored_moves'][:size_array[j], 1]]
        cur_start_index += size_array[j]

    return piece_bbs, occupied_bbs, ep_squares, from_to_squares, size_array


def start_move_scoring(children, not_children, children_indices, not_children_indices, move_eval_fn):
    child_best_move_scores = np.zeros(len(children_indices), dtype=np.float32)
    not_child_best_move_scores = np.zeros(len(not_children_indices), dtype=np.float32)

    def set_move_scores():
        for_ann = move_scoring_helper(children, not_children, children_indices, not_children_indices)
        results = move_eval_fn(*for_ann)

        jitted_assign_move_scores(children, not_children, children_indices, not_children_indices, for_ann[-1], results,
                                  child_best_move_scores, not_child_best_move_scores)

    t = threading.Thread(target=set_move_scores)
    t.start()
    return t, child_best_move_scores, not_child_best_move_scores


@njit
def set_evaluation_scores(struct_array, to_set_mask, results):
    index_in_results = 0
    for j in range(len(struct_array)):
        if to_set_mask[j]:
            struct_array[j]['best_value'] = results[index_in_results]
            index_in_results += 1





def start_board_evaluations(struct_array, to_score_mask, board_eval_fn):
    """
    Start the evaluation of the depth zero nodes which were not previously terminated, and set their results to their
    best_value field.

    :return: The thread responsible for evaluating and setting the struct's best_value fields to the results
    """

    def evaluate_and_set():
        results = board_eval_fn(
            *struct_array_to_ann_input_all_white(
                struct_array[to_score_mask])).squeeze(axis=1)
        set_evaluation_scores(struct_array, to_score_mask, results)

    t = threading.Thread(target=evaluate_and_set)

    t.start()
    return t


@njit
def has_insufficient_material(board_state):
    # Enough material to mate.
    if board_state.pawns or board_state.rooks or board_state.queens:
        return False

    # A single knight or a single bishop.
    if popcount(board_state.occupied) <= 3:
        return True

    # More than a single knight.
    if board_state.knights:
        return False

    # All bishops on the same color.
    if board_state.bishops & BB_DARK_SQUARES == 0:
        return True
    elif board_state.bishops & BB_LIGHT_SQUARES == 0:
        return True
    else:
        return False


@njit
def struct_terminated_from_tt(board_struct, hash_table):
    """
    Checks if the node should be terminated from the information contained in the given hash_table.  It also
    updates the values in the given node when applicable.

    NOTES:
    1) While this currently does work, it represents one of the most crucial components of the negamax search,
    and has not been given the thought and effort it needs.  This is an extremely high priority
    """
    hash_entry = hash_table[board_struct.hash & ONES_IN_RELEVANT_BITS_FOR_TT_INDEX]
    if hash_entry["depth"] != NO_TT_ENTRY_VALUE:
        if hash_entry["entry_hash"] == board_struct.hash:
            if hash_entry["depth"] >= board_struct.depth:
                if hash_entry["lower_bound"] >= board_struct.separator: #This may want to be >
                    board_struct.best_value = hash_entry["lower_bound"]
                    return True
                else:
                    if hash_entry["lower_bound"] > board_struct.best_value:
                        board_struct.best_value = hash_entry["lower_bound"]
                    if hash_entry["upper_bound"] < board_struct.separator:  #This may want to be <=
                        board_struct.best_value = hash_entry["upper_bound"]
                        return True
    return False




@njit
def depth_zero_should_terminate_array(struct_array, hash_table):
    """
    This function goes through the given struct_array and looks for depth zero nodes which should terminate
    for reasons other than scoring by the evaluation function.  This is separate from
    the get_indices_of_terminating_children function so that it can give the GPU the boards for evaluation as quickly
    as possible.


    CURRENTLY CHECKING:
    1) Draw by the 50-move rule
    2) Draw by insufficient material
    3) Draw by stalemate
    4) Win/loss by checkmate
    5) Termination by information contained in the TT

    MUST IMPLEMENT:
    1) Draw by threefold repetition
    """
    for j in range(len(struct_array)):
        if struct_array[j]['depth'] == 0:
            if struct_array[j]['halfmove_clock'] >= 50 or has_insufficient_material(struct_array[j]):
                struct_array[j]['terminated'] = True
                struct_array[j]['best_value'] = TIE_RESULT_SCORE
            elif struct_terminated_from_tt(struct_array[j], hash_table):
                struct_array[j]['terminated'] = True
            else:
                TEMP_JITCLASS_OBJECT = BoardState(struct_array[j]['pawns'],   ###############THIS MUST BE REMOVED.  IT IS THE LAST REMAINING CONVERSION TO BoardState objects
                                                  struct_array[j]['knights'],
                                                  struct_array[j]['bishops'],
                                                  struct_array[j]['rooks'],
                                                  struct_array[j]['queens'],
                                                  struct_array[j]['kings'],
                                                  struct_array[j]['occupied_w'],
                                                  struct_array[j]['occupied_b'],
                                                  struct_array[j]['occupied'],
                                                  struct_array[j]['turn'],
                                                  struct_array[j]['castling_rights'],
                                                  None if struct_array[j]['ep_square'] == 0 else struct_array[j]['ep_square'],
                                                  struct_array[j]['halfmove_clock'],
                                                  struct_array[j]['hash'])

                ###########This can and should be removed since the information is already computed during the move generation.
                if not has_legal_move(TEMP_JITCLASS_OBJECT):

                    struct_array[j]['terminated'] = True
                    if is_in_check(TEMP_JITCLASS_OBJECT):
                        struct_array[j]['best_value'] = LOSS_RESULT_SCORE
                    else:
                        struct_array[j]['best_value'] = TIE_RESULT_SCORE



@njit
def has_legal_tt_move(board_struct, hash_table):
    """
    Checks if a move is being stored in the transposition table for the given board struct, and if there is, that the
    move is legal.  If it does find a legal move, it stores the move in the struct's first move index, sets the
    struct's next move score to a constant value, and sets it's children_left to a specified constant to indicate there
    was a legal move found in the tt.

    :return: True if a move is found, or False if not.
    """
    node_entry = hash_table[board_struct['hash'] & ONES_IN_RELEVANT_BITS_FOR_TT_INDEX]
    if node_entry['depth'] != NO_TT_ENTRY_VALUE:
        if node_entry['entry_hash'] == board_struct['hash']:
            if node_entry['stored_from_square'] != NO_TT_MOVE_VALUE:
                move_tuple = (node_entry["stored_from_square"],
                                  node_entry["stored_to_square"],
                                  node_entry["stored_promotion"])

                if scalar_is_legal_move(board_struct, move_tuple):
                    board_struct['unexplored_moves'][0] = move_tuple
                    board_struct['next_move_index'] = 0
                    board_struct['children_left'] = NEXT_MOVE_IS_FROM_TT_VAL
                    return True

    return False




@njit
def get_indices_of_terminating_children(struct_array, hash_table):
    """
    CURRENTLY CHECKING:
    1) Draw by the 50-move rule
    2) Draw by insufficient material
    3) Draw by stalemate
    4) Win/loss by checkmate
    5) Termination by information contained in the TT

    MUST IMPLEMENT:
    1) Draw by threefold repetition
    """
    for j in range(len(struct_array)):
        if struct_array[j]['depth'] != 0:
            if struct_array[j]["halfmove_clock"] >= 50 or has_insufficient_material(struct_array[j]):
                struct_array[j]['best_value'] = TIE_RESULT_SCORE
                struct_array[j]['terminated'] = True
            elif struct_terminated_from_tt(struct_array[j], hash_table):
                struct_array[j]['terminated'] = True
            elif has_legal_tt_move(struct_array[j], hash_table):
                pass
            else:
                set_up_move_array(struct_array[j])



@njit
def create_child_structs(struct_array, testing=False):
    #This should not copy the entire array, instead only the fields which are not directly written over
    child_array = struct_array.copy()

    # Will use np.empty_like when more confident in the implementation
    new_next_move_values = np.zeros_like(struct_array['best_value'])

    #EVENTUALLY THIS NEEDS TO BE REMOVED AND THE prev_move OF child_array SHOULD BE USED INSTEAD
    moves_to_push = np.zeros((len(struct_array), 3), dtype=np.uint8)

    for j in range(len(struct_array)):
        child_array[j]['unexplored_moves'][:] = 255
        child_array[j]['unexplored_move_scores'][:] = MIN_FLOAT32_VAL
        child_array[j]['prev_move'][:] = struct_array[j]['unexplored_moves'][struct_array[j]['next_move_index']]
        child_array[j]['depth'] = struct_array[j]['depth'] -1

        moves_to_push[j] = struct_array[j]['unexplored_moves'][struct_array[j]['next_move_index']]

        if testing:
            if np.any(child_array[j]['prev_move'][:]==255):
                print("THE FOLLOWING MOVE IS ATTEMPTING TO BE PUSHED:", child_array[j]['prev_move'])

        if struct_array[j]['children_left'] != NEXT_MOVE_IS_FROM_TT_VAL:
            new_next_move_values[j] = set_up_next_best_move(struct_array[j])
        else:
            new_next_move_values[j] = TT_MOVE_SCORE_VALUE

    push_moves(child_array, moves_to_push)


    child_array['best_value'][:] = MIN_FLOAT32_VAL
    child_array['children_left'][:] = 0
    child_array['separator'][:] *= -1
    child_array['next_move_index'][:] = 255

    return child_array, new_next_move_values


@njit
def generate_moves_for_tt_move_nodes(struct_array, to_check_mask):
    """
    Generates the legal moves for the array of board structs when a legal move for the node was found in
    the transposition table (TT), and then was expanded in the same iteration as this function is being run.
    It generates all of the legal moves except for the move which was already found in the TT.
    It then sets each struct's children_left to the actual number of children left,
    as opposed to the indicator value NEXT_MOVE_IS_FROM_TT_VAL which it previously was.
    This function returns a mask for the structs which have more unexplored children to explore.
    """
    for j in range(len(struct_array)):
        if to_check_mask[j]:
            struct_array[j]['children_left'] = 0
            set_up_move_array_except_move(struct_array[j], struct_array[j]['unexplored_moves'][0].copy())
            struct_array[j]['children_left'] += 1

    return struct_array[to_check_mask]['children_left'] != 1


def create_nodes_for_struct(struct_array, parent_nodes):
    to_return = [None for _ in range(len(struct_array))]
    for j in range(len(struct_array)):
        to_return[j] = GameNode(np.array([struct_array[j]], dtype=numpy_node_info_dtype), parent_nodes[j])
    return to_return


def get_struct_array_from_node_array(node_array):
    return np.array([node.board_struct[0] for node in node_array],dtype=numpy_node_info_dtype)


@njit
def update_node_from_value(node, value, following_move, hash_table):
    if not node is None:
        if not node.board_struct[0]['terminated']:  #This may be preventing more accurate predictions when two+ nodes which would terminate a single node in one batch iteration are updated out of order (see comment of update_tree_from_terminating_nodes)
            node.board_struct[0]['children_left'] -= 1
            node.board_struct[0]['best_value'] = max(value, node.board_struct[0]['best_value'])

            if node.board_struct[0]['best_value'] >= node.board_struct[0]['separator'] or node.board_struct[0]['children_left'] == 0:
                node.board_struct[0]['terminated'] = True
                add_board_and_move_to_tt(node.board_struct[0], following_move, hash_table)

                # This is being checked at the start of the call also and thus should be removedMoved several dependencies to newer versions
                if not node.parent is None:
                    update_node_from_value(node.parent, - node.board_struct[0]['best_value'], node.board_struct[0]['prev_move'], hash_table)


@njit
def temp_jitted_start_tree_update_from_node(parent_node, child_struct, hash_table):
    update_node_from_value(parent_node,
                           - child_struct['best_value'],
                           child_struct['prev_move'],
                           hash_table)


def update_tree_from_terminating_nodes(parent_nodes, struct_array, hash_table, testing=False):
    """
    Updates the search tree from the nodes in the current batch which are terminating, this includes all nodes which
    have been marked terminated, or are depth zero.  It also updates the transposition table as needed.

    EXTREMELY IMPORTANT NODES:
    1) If a node terminates one of it's parent's which would also have be terminated by another terminating node
    from the same batch, then the parent will terminate with the value of whichever node was first in this
    methods for loop.  This is obviously not desired, as the higher value for termination should be the one to
    terminate the parent. If this proves to greatly effect the results of the search or it's speed, it will need to
    sort the given nodes to update, or keep track of the nodes which were terminated in the current batch.
    """
    should_update_mask = np.logical_or(struct_array['depth'] == 0, struct_array['terminated'])

    if testing:
        if np.any(np.logical_or(struct_array[should_update_mask]['best_value'] == MIN_FLOAT32_VAL,
                                struct_array[should_update_mask]['best_value'] == MAX_FLOAT32_VAL)):
            print("The tree is about to be updated with an value which should be unreachable!")


    # Since no node being given to this function will have terminated from a child, none of them will have moves to store in the TT.
    # Also the passing of a copy instead of a view will (I think) ensure the memory be aligned for compilation to
    # vectorized versions of the looped functions
    # add_boards_to_tt(struct_array[should_update_mask],hash_table)

    for j in range(len(struct_array)):
        if should_update_mask[j]:
            temp_jitted_start_tree_update_from_node(parent_nodes[j], struct_array[j], hash_table)



@njit
def jitted_temp_set_node_from_altered_struct(node, board_struct):
    """
    NOTES:
    1) This is VERY slow, and it's removal is a top priority.
    2) Sometimes this is called when no values actually differ, I have a fix for this that seems to work on every
    test I try, but the reasoning as to why it works isn't good enough for me to trust it completely.  Finishing that
    up is likely one of the next things to be done.
    """
    node.board_struct[0]['unexplored_moves'][:] = board_struct['unexplored_moves'][:]
    node.board_struct[0]['unexplored_move_scores'][:] = board_struct['unexplored_move_scores'][:]
    node.board_struct[0]['children_left'] = board_struct['children_left']
    node.board_struct[0]['next_move_index'] = board_struct['next_move_index']



@njit
def jitted_temp_set_node_move_score_info(node, board_struct):
    """
    NOTES:
    1) This is VERY slow, and it's removal is a top priority.
    2) Sometimes this is called when no values actually differ, I have a fix for this that seems to work on every
    test I try, but the reasoning as to why it works isn't good enough for me to trust it completely.  Finishing that
    up is likely one of the next things to be done.
    """
    node.board_struct[0]['unexplored_move_scores'][:] = board_struct['unexplored_move_scores'][:]
    node.board_struct[0]['next_move_index'] = board_struct['next_move_index']



def temp_set_nodes_to_altered_structs(node_array, struct_array):
    """
    NOTES:
    1) This is VERY slow, and it's removal is a top priority. Also some of it's assignments are just unneeded.
    """
    for j in range(len(node_array)):
        jitted_temp_set_node_from_altered_struct(node_array[j], struct_array[j])




def temp_set_tt_move_scores(node_array, struct_array):
    """
    NOTES:
    1) This is VERY slow, and it's removal is a top priority.
    """
    for j in range(len(node_array)):
        jitted_temp_set_node_move_score_info(node_array[j], struct_array[j])



def do_iteration(node_batch, hash_table, board_eval_fn, move_eval_fn, testing=False):
    struct_batch = get_struct_array_from_node_array(node_batch)
    child_struct, struct_batch_next_move_scores = create_child_structs(struct_batch, testing)

    child_was_from_tt_move_mask = struct_batch['children_left'] == NEXT_MOVE_IS_FROM_TT_VAL

    depth_zero_children_mask = child_struct['depth'] == 0
    depth_not_zero_mask = np.logical_not(depth_zero_children_mask)

    depth_zero_should_terminate_array(child_struct, hash_table)


    depth_zero_not_scored_mask = np.logical_and(depth_zero_children_mask, np.logical_not(child_struct['terminated']))


    if testing:
        if np.any(child_struct[np.logical_and(depth_zero_children_mask, child_struct['terminated'])]['best_value']==MIN_FLOAT32_VAL):
            print("A depth zero node was marked as terminated without a value for termination prior to evaluations!")


    if np.any(depth_zero_not_scored_mask):
        evaluation_thread = start_board_evaluations(child_struct, depth_zero_not_scored_mask, board_eval_fn)
    else:
        evaluation_thread = None


    #(This should likely be put directly into it's only use below to prevent storing it in memory)
    tt_move_nodes_with_more_kids_indices = generate_moves_for_tt_move_nodes(struct_batch, child_was_from_tt_move_mask)

    SUPER_TEMP_MASK = np.logical_not(np.logical_and(child_was_from_tt_move_mask, struct_batch['children_left'] == 1))


    get_indices_of_terminating_children(child_struct, hash_table)


    non_zero_depth_child_not_term_indices = np.logical_and(
        depth_not_zero_mask,
        np.logical_not(child_struct['terminated']))


    found_no_move_in_tt_indices = np.logical_and(
        non_zero_depth_child_not_term_indices,
        child_struct['children_left'] != NEXT_MOVE_IS_FROM_TT_VAL)

    to_score_child_moves_indices = np.arange(len(child_struct))[found_no_move_in_tt_indices]
    to_score_not_child_moves_indices = np.arange(len(struct_batch))[child_was_from_tt_move_mask][tt_move_nodes_with_more_kids_indices]


    if len(to_score_child_moves_indices) != 0 or len(to_score_not_child_moves_indices) != 0:
        move_thread, child_next_move_scores, not_child_next_move_scores = start_move_scoring(
            child_struct,
            struct_batch,
            to_score_child_moves_indices,
            to_score_not_child_moves_indices,
            move_eval_fn)
    else:
        move_thread = None
        child_next_move_scores = None
        not_child_next_move_scores = None


    # A mask of the given batch which have more unexplored children left
    have_children_left_mask = np.logical_and(
        struct_batch["next_move_index"] != 255,
        SUPER_TEMP_MASK)


    temp_set_nodes_to_altered_structs(
        node_batch[have_children_left_mask],
        struct_batch[have_children_left_mask])

    if not evaluation_thread is None:
        evaluation_thread.join()

    update_tree_from_terminating_nodes(
        node_batch,
        child_struct,
        hash_table,
        testing=testing)



    if not move_thread is None:
        move_thread.join()

        # Set the boards who used TT moves to generate a child move scores in their respective nodes
        temp_set_tt_move_scores(node_batch[to_score_not_child_moves_indices],struct_batch[to_score_not_child_moves_indices])



    new_child_nodes = create_nodes_for_struct(
        child_struct[non_zero_depth_child_not_term_indices],
        node_batch[non_zero_depth_child_not_term_indices])

    to_insert = np.concatenate((
        node_batch[have_children_left_mask],
        new_child_nodes))


    #Set up the array of scores used to place the returned nodes into their proper bins
    scores_to_return = np.full(len(to_insert), TT_MOVE_SCORE_VALUE, dtype=np.float32)
    num_not_child_scores = len(to_insert) - len(new_child_nodes)
    if num_not_child_scores != 0:
        scores_to_return[:num_not_child_scores] = struct_batch_next_move_scores[have_children_left_mask]
        if not not_child_next_move_scores is None and len(not_child_next_move_scores) != 0:
            scores_to_return[:num_not_child_scores][scores_to_return[:num_not_child_scores] == TT_MOVE_SCORE_VALUE] = not_child_next_move_scores
    if not child_next_move_scores is None and len(child_next_move_scores) != 0:
            scores_to_return[num_not_child_scores:][child_struct[non_zero_depth_child_not_term_indices]['children_left'] != NEXT_MOVE_IS_FROM_TT_VAL] = child_next_move_scores


    return to_insert, scores_to_return




def do_alpha_beta_search_with_bins(root_game_node, max_batch_size, bins_to_use, board_eval_fn, move_eval_fn, hash_table=None, print_info=False, full_testing=False):
    open_node_holder = PriorityBins(bins_to_use, max_batch_size, testing=full_testing) #This will likely be put outside of this function long term

    if hash_table is None:
        hash_table = get_empty_hash_table()

    next_batch = np.array([root_game_node], dtype=np.object)

    if print_info or full_testing:
        iteration_counter = 0
        total_nodes_computed = 0
        given_for_insert = 0
        start_time = time.time()

    while len(next_batch) != 0:
        if print_info or full_testing:
            num_open_nodes = len(open_node_holder)
            print("Starting iteration", iteration_counter, "with batch size of", len(next_batch), "and", num_open_nodes, "open nodes, in", open_node_holder.num_non_empty(),"non-empty bins, with a max bin of", open_node_holder.largest_bin())
            total_nodes_computed += len(next_batch)

            if full_testing:
                if root_game_node.board_struct[0]['terminated'] or root_game_node.board_struct[0]['best_value'] > root_game_node.board_struct[0]['separator'] or root_game_node.board_struct[0]['children_left'] == 0:
                    print("ROOT NODE WAS TERMINATED OR SHOULD BE TERMINATED AT START OF ITERATION.")

        to_insert, to_insert_scores = do_iteration(next_batch, hash_table, board_eval_fn, move_eval_fn, full_testing)


        if print_info or full_testing:
            given_for_insert += len(to_insert)

            if full_testing:
                if len(to_insert) != len(to_insert_scores):
                    print("The number of nodes from the previous iteration doesn't equal the number of scores received!")
                for j in range(len(to_insert)):
                    if to_insert[j] is None:
                        print("Found none in array being inserted into global nodes!")
                    if to_insert[j].board_struct[0]['depth'] == 0:
                        print("Found a node with depth zero being inserted into global nodes!")

                if root_game_node.board_struct[0]['terminated'] or root_game_node.board_struct[0]['best_value'] > root_game_node.board_struct[0]['separator'] or root_game_node.board_struct[0]['children_left'] == 0:
                    print("ROOT NODE WAS TERMINATED OR SHOULD BE TERMINATED BEFORE INSERTING NODES INTO BINS")

        if print_info or full_testing:
            print("Iteration", iteration_counter, "completed producing", len(to_insert), "nodes to insert.")


        if len(to_insert) != 0 or not open_node_holder.is_empty():

            if full_testing:
                if len(to_insert) != len(to_insert_scores):
                    print("do_iteration returned %d nodes to insert, and %d scores"%(len(to_insert), len(to_insert_scores)))

            next_batch = open_node_holder.insert_nodes_and_get_next_batch(to_insert, to_insert_scores)
        else:
            next_batch = []

        if print_info or full_testing:
            iteration_counter += 1

            if full_testing:
                for j in range(len(next_batch)):
                    if next_batch[j] is None:
                        print("Found None in next_batch!")
                    elif next_batch[j].board_struct[0]['terminated']:
                        print("A terminated node was found in the newly created next_batch!")

    if print_info or full_testing:
        time_taken_without_tt = time.time() - start_time
        print("Total time taken not including table_creation:", time_taken_without_tt)
        print("Number of iterations taken", iteration_counter, "in total time", time_taken_without_tt)
        print("Average time per iteration:", time_taken_without_tt / iteration_counter)
        print("Total nodes computed:", total_nodes_computed)
        print("Nodes evaluated per second:", total_nodes_computed/time_taken_without_tt)
        print("Total nodes given to bin arrays for insert", given_for_insert)

    return root_game_node.board_struct['best_value']





def create_root_game_node_from_fen(fen, depth=255, seperator=0):
    return GameNode(create_node_info_from_fen(fen, depth, seperator), None)


def set_up_root_node(move_eval_fn, fen, depth=255, separator=0):
    root_node = create_root_game_node_from_fen(fen, depth, separator)
    set_up_move_array(root_node.board_struct[0])

    thread, _, _ =  start_move_scoring(
        root_node.board_struct,
        root_node.board_struct, #This isn't used but is needed to run
        np.arange(1, dtype=np.int32),
        np.arange(0, dtype=np.int32),
        move_eval_fn)

    thread.join()
    set_up_next_best_move(root_node.board_struct[0])
    return root_node




def mtd_f(fen, depth, first_guess, min_window_to_confirm, board_eval_fn, move_eval_fn, guess_increment=.5, search_batch_size=1000, bins_to_use=None, hash_table=None, win_threshold=1000000, loss_threshold=-1000000, print_info=False, full_testing=False):
    if hash_table is None:
        hash_table = get_empty_hash_table()

    if bins_to_use is None:
        bins_to_use = np.arange(15, -15, -.025)


    counter = 0
    cur_guess = first_guess
    upper_bound = MAX_FLOAT32_VAL
    lower_bound = MIN_FLOAT32_VAL
    while upper_bound -  min_window_to_confirm > lower_bound:
        if upper_bound == MAX_FLOAT32_VAL and lower_bound >= win_threshold:
            print("BREAKING MTD(F) LOOP FROM WIN THRESHOLD")
            break

        if lower_bound == MIN_FLOAT32_VAL and upper_bound <= loss_threshold:
            print("BREAKING MTD(F) LOOP FROM LOSS THRESHOLD")
            break

        if lower_bound == MIN_FLOAT32_VAL:
            if upper_bound != MAX_FLOAT32_VAL:
                beta = upper_bound - guess_increment
            else:
                beta = cur_guess
        elif upper_bound == MAX_FLOAT32_VAL:
            beta = lower_bound + guess_increment
        else:
            beta = lower_bound + (upper_bound - lower_bound) / 2

        #This would ideally share the same tree, but updated for the new seperation value
        cur_root_node = set_up_root_node(move_eval_fn, fen, depth, beta - 1000*np.finfo(np.float32).eps)

        cur_guess = do_alpha_beta_search_with_bins(cur_root_node, search_batch_size, bins_to_use, board_eval_fn, move_eval_fn, hash_table=hash_table, print_info=print_info, full_testing=full_testing)
        if cur_guess < beta:
            upper_bound = cur_guess
        else:
            lower_bound = cur_guess

        if print_info:
            print("Finished iteration %d with lower and upper bounds (%f,%f)" % (counter+1, lower_bound, upper_bound))

        counter += 1


    root_tt_entry = hash_table[np.uint64(cur_root_node.board_struct[0]['hash']) & ONES_IN_RELEVANT_BITS_FOR_TT_INDEX]
    try:
        tt_move = chess.Move(root_tt_entry["stored_from_square"],
                       root_tt_entry["stored_to_square"],
                       root_tt_entry["stored_promotion"])
    except IndexError as E:
        tt_move = None
        print("FAILED TO CREATE MOVE in MTD(f)")
        raise(E)

    return cur_guess, tt_move, hash_table





def iterative_deepening_mtd_f(fen, depths_to_search, batch_sizes, min_windows_to_confirm, board_eval_fn, move_eval_fn, first_guess=0, guess_increments=None, bin_sets=None, hash_table=None, win_threshold=1000000, loss_threshold=-1000000, print_info=False, full_testing=True):
    if hash_table is None:
        hash_table = get_empty_hash_table()

    if bin_sets is None:
        bin_sets = (np.arange(15, -15, -.025) for _ in range(len(depths_to_search)))

    if guess_increments is None:
        guess_increments = [5]*len(depths_to_search)

    for depth, batch_size, window, increment, bins in zip(depths_to_search, batch_sizes, min_windows_to_confirm, guess_increments, bin_sets):
        # print("Starting depth %d search"%depth)
        start_time = time.time()
        first_guess, tt_move, hash_table = mtd_f(
            fen,
            depth,
            first_guess,
            window,
            board_eval_fn,
            move_eval_fn,
            guess_increment=increment,
            search_batch_size=batch_size,
            bins_to_use=bins,
            hash_table=hash_table,
            win_threshold=win_threshold,
            loss_threshold=loss_threshold,
            print_info=print_info,
            full_testing=full_testing)


        if print_info:
            print("Completed depth %d in time %f"%(depth, time.time() - start_time))


    return first_guess, tt_move, hash_table






if __name__ == "__main__":

    temp_piece_values = np.array([900,500,300,300,100,500],dtype=np.float32)
    @njit(nogil=True)
    def piece_sum_eval(pieces, occupied_bbs, unused):
        piece_counts = vectorized_popcount(np.bitwise_and(pieces[...,1:], occupied_bbs))
        return np.sum(temp_piece_values*(piece_counts[:,0].view(np.int8) - piece_counts[:,1].view(np.int8)).astype(np.float32),axis=1)

    @njit(nogil=True)
    def random_move_eval(unused_1, unused_2, unused_3, unused_4, num_moves_per_board):
        return np.linspace(-1,1,np.sum(num_moves_per_board)) #it's not really random
        # return np.random.rand(np.sum(num_moves_per_board))




        




    FEN_TO_TEST = "r2q1rk1/pbpnbppp/1p2pn2/3pN3/2PP4/1PN3P1/P2BPPBP/R2Q1RK1 w - - 10 11"#"rn1qkb1r/p1pp1ppp/bp2pn2/8/2PP4/5NP1/PP2PP1P/RNBQKB1R w KQkq - 1 5"###"rn1qkb1r/p1pp1ppp/bp2pn2/8/2PP4/5NP1/PP2PP1P/RNBQKB1R w KQkq - 1 5"#"#"rn1q1rk1/pbp1bppp/1p2pn2/3pN3/2PP4/1P4P1/P2BPPBP/RN1Q1RK1 w - - 8 10"#"  #
    DEPTH_OF_SEARCH = 5
    MAX_BATCH_LIST =  [5000]*4#[10000, 10000, 5000,2500]#,1000,500,100]#[5000, 2000]#, 5000, 2000, 1000, 723, 500,1]# 5*[10000]#[5000]*3#[5000, 2000, 500]#,1,2,3,4,5,6,7,20]#[1,2,3,4]##1*[250,500, 1000, 2000]# [j**2 for j in range(40,0,-2)] + [5000]
    SEPERATING_VALUE = 0
    BINS_TO_USE = np.arange(20, -20, -.01)
    # BINS_TO_USE = np.arange(4000, -4000, 99.99)
    BOARD_EVAL_GRAPHDEF_FILE = "/srv/tmp/encoder_evaluation/conv_train_wide_and_deep_4/1526978123/tensorrt_eval_graph.pb"


    PREDICTOR, _, CLOSER = get_inference_functions(BOARD_EVAL_GRAPHDEF_FILE, None)
    MOVE_PREDICTOR, MOVE_CLOSER = move_ann_main([True])


    temp_board_to_test = chess.Board(FEN_TO_TEST)
    print(temp_board_to_test)
    for batch_size in MAX_BATCH_LIST:
        for first_guess in [0]:
            start_time = time.time()
            results = mtd_f(FEN_TO_TEST,DEPTH_OF_SEARCH,first_guess,.001, PREDICTOR, MOVE_PREDICTOR, search_batch_size=batch_size, print_info=True, full_testing=False)
            # results = mtd_f(FEN_TO_TEST, DEPTH_OF_SEARCH, first_guess, 99, piece_sum_eval, random_move_eval,guess_increment=98,search_batch_size=batch_size, print_info=True, full_testing=False)
            print(results[0], results[1] ,"found by mtd(f)", time.time() - start_time, "\n")

            # depths_to_check = np.arange(0,DEPTH_OF_SEARCH,2) + 1
            # results = iterative_deepening_mtd_f(
            #     FEN_TO_TEST,
            #     depths_to_check,
            #     [10000 for _ in range(len(depths_to_check))],
            #     [.001 for _ in range(len(depths_to_check))],
            #     PREDICTOR,
            #     MOVE_PREDICTOR,
            #     bin_sets=[BINS_TO_USE for _ in range(len(depths_to_check))],
            # )

    CLOSER()
    MOVE_CLOSER()

