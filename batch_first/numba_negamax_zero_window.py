import threading
import time

from .numba_board import *
from . import transposition_table as tt

from .classes_and_structs import *



@njit
def compress_square_array(to_compress):
    """
    Given an array of uint8s which all have values less than 16, compress them by storing
    two values per uint8 (as opposed to one).
    """
    to_compress[::2] <<= np.uint8(4)
    to_compress[::2] |= to_compress[1::2]
    return to_compress[::2]


@njit
def square_scanner_helper(bb, mirror_squares=False):
    if mirror_squares:
        for square in scan_reversed(bb):
            yield square_mirror(square)
    else:
        for square in scan_reversed(bb):
            yield square


@njit
def get_square_ary(struct, relevent_square_mask):
    """
    Creates and returns an array of the ann filters indices corresponding to the given boards 'relevant' squares.
    A 'relevant' square is a square that's either occupied or an ep-capture square.
    """
    squares = np.empty(popcount(relevent_square_mask), np.uint8)
    for j, square in enumerate(square_scanner_helper(relevent_square_mask, not struct['turn'])):
        type = piece_type_at(struct, square)

        if type:
            bb_square = BB_SQUARES[square]
            squares[j] = 7 if struct['occupied_co'][struct['turn']] & bb_square else 14
            if struct['castling_rights'] & bb_square == 0:
                squares[j] -= type
        else:
            squares[j] = 0

    return squares


@njit
def own_concat(lst, total_size):
    """
    A customized implementation of np.concatenate (mainly used because Numba wouldn't let it use np.concatenate for some reason)
    """
    to_return = np.empty(total_size + (total_size % 2), np.uint8)
    start_index = 0
    for to_place in lst:
        end_index = start_index + len(to_place)
        to_return[start_index:end_index] = to_place
        start_index = end_index

    if total_size % 2: #to fix the issues caused by part of the array staying 'empty'
        to_return[-1] = 0

    return to_return


@njit
def struct_array_to_ann_inputs(child_structs, not_child_structs, to_score_child_mask, to_score_not_child_mask, total_num_to_score):
    """
    Uses ceil((popcnt(occupied)+int(has_ep))/2)*8+64=[80, 200] bits per board transferred.
    """
    occupied_bbs = np.empty(total_num_to_score, dtype=np.uint64)

    total_squares = 0
    store_index = 0
    to_concat = []
    for j in range(len(child_structs) + len(not_child_structs)):
        if j < len(child_structs):
            should_score = to_score_child_mask[j]
            struct = child_structs[j]
        else:
            should_score = to_score_not_child_mask[j - len(child_structs)]
            struct = not_child_structs[j - len(child_structs)]

        if should_score:
            occupied_bbs[store_index] = struct['occupied']
            if struct['ep_square']:
                occupied_bbs[store_index] |= BB_SQUARES[struct['ep_square']]

            if not struct['turn']:
                occupied_bbs[store_index] = flip_vertically(occupied_bbs[store_index])

            cur_squares = get_square_ary(struct, occupied_bbs[store_index])

            total_squares += len(cur_squares)
            to_concat.append(cur_squares)

            store_index += 1
    compressed_squares = compress_square_array(own_concat(to_concat, total_squares))

    return compressed_squares, occupied_bbs


@njit
def can_draw_from_repetition(board_struct, parent_node, previous_board_map):
    """
    NOTES:
     1) Look into if hash collisions are something thing I need to be checking for (type 1 collisions)
    """
    if board_struct['halfmove_clock'] < 4:
        return False

    #Checks if the board is a repitition of a board which was made in the actual game (meaning prior to the current search)
    hash_index = np.searchsorted(previous_board_map[0], board_struct['hash'])
    if board_struct['hash'] == previous_board_map[hash_index, 0]:
        num_found_so_far = 1 + previous_board_map[hash_index, 1]
    else:
        num_found_so_far = 1


    if parent_node.parent is None:
        return False

    node = parent_node.parent

    if board_struct['hash'] == node.struct['hash']:
        num_found_so_far += 1

    if num_found_so_far >= 3:
        return True

    for num in range(node.struct['halfmove_clock'], 1, -2):
        if num < 3 and num_found_so_far == 1:
            return False
        elif node.parent is None or node.parent.parent is None:
            return False

        node = node.parent.parent

        if board_struct['hash'] == node.struct['hash']:
            num_found_so_far += 1

            if num_found_so_far >= 3:
                return True

    return False


@njit
def set_up_next_best_move(board_struct):
    board_struct['next_move_index'] = np.argmax(board_struct['unexplored_move_scores'])
    best_move_score = board_struct['unexplored_move_scores'][board_struct['next_move_index']]
    if best_move_score == MIN_FLOAT32_VAL:
        board_struct['next_move_index'] = NO_MORE_MOVES_VALUE
    else:
        board_struct['unexplored_move_scores'][board_struct['next_move_index']] = MIN_FLOAT32_VAL
    return best_move_score


def start_move_scoring(children, not_children, child_score_mask, not_child_score_mask, move_eval_fn):
    num_children_to_score = np.sum(child_score_mask)
    num_not_child_to_score = np.sum(not_child_score_mask)

    result_getter = [None]
    def set_move_scores():
        result_getter[0] = move_eval_fn(
            *struct_array_to_ann_inputs(
                children,
                not_children,
                child_score_mask,
                not_child_score_mask,
                num_children_to_score + num_not_child_to_score))

    t = threading.Thread(target=set_move_scores)
    t.start()
    return t, result_getter, num_children_to_score, num_not_child_to_score


def start_board_evaluations(struct_array, to_score_mask, board_eval_fn):
    """
    Start the evaluation of the depth zero nodes which were not previously terminated.
    """
    num_to_score = np.sum(to_score_mask)
    evaluation_scores = np.empty(num_to_score, dtype=np.float32)

    def evaluate_and_set():
        evaluation_scores[:] = board_eval_fn(
            *struct_array_to_ann_inputs(
                struct_array,
                np.array([], dtype=numpy_node_info_dtype),
                to_score_mask,
                np.array([], dtype=np.bool_),
                num_to_score))

    t = threading.Thread(target=evaluate_and_set)
    t.start()

    return t, evaluation_scores


@njit
def should_terminate_from_tt(board_struct, hash_table):
    """
    Checks if the node should be terminated from the information contained in the given hash_table.  It also
    updates the values in the given node when applicable.
    """
    hash_entry = hash_table[board_struct['hash'] & TT_HASH_MASK]
    if hash_entry['depth'] != NO_TT_ENTRY_VALUE:
        if hash_entry['entry_hash'] == board_struct['hash']:
            if hash_entry['depth'] >= board_struct['depth']:
                if hash_entry['lower_bound'] >= board_struct['separator']:
                    board_struct['best_value'] = hash_entry['lower_bound']
                    return True
                else:
                    if hash_entry['upper_bound'] < board_struct['separator']:
                        board_struct['best_value'] = hash_entry['upper_bound']
                        return True
                    if hash_entry['lower_bound'] > board_struct['best_value']:
                        board_struct['best_value'] = hash_entry['lower_bound']
    return False


@njit
def depth_zero_should_terminate_array(struct_array, hash_table, previous_board_map, node_holder):
    """
    This function goes through the given struct_array and looks for depth zero nodes which should terminate
    for reasons other than scoring by the evaluation function.  This is separate from
    the child_termination_check_and_move_gen function so that it can give the GPU the boards for evaluation as quickly
    as possible.


    Things being checked:
    1) Draw by the 50-move rule
    2) Draw by insufficient material
    3) Draw by stalemate
    4) Win/loss by checkmate
    5) Termination by information contained in the TT
    6) Draw by threefold repetition
    """
    for j in range(len(struct_array)):
        if struct_array[j]['depth'] == 0:
            if struct_array[j]['halfmove_clock'] >= 50 or has_insufficient_material(struct_array[j]):
                struct_array[j]['terminated'] = True
                struct_array[j]['best_value'] = TIE_RESULT_SCORE
            elif can_draw_from_repetition(struct_array[j], node_holder.held_node, previous_board_map):
                # This is just assigning a draw value, though it doesn't necessarily imply a draw,
                # just that one can be claimed.   Not sure if this needs to be handled, and if yes how to handle it
                struct_array[j]['terminated'] = True
                struct_array[j]['best_value'] = TIE_RESULT_SCORE
            elif should_terminate_from_tt(struct_array[j], hash_table):
                struct_array[j]['terminated'] = True
            elif not has_legal_move(struct_array[j]):
                struct_array[j]['terminated'] = True

        node_holder = node_holder.next_holder


@njit
def has_legal_tt_move(board_struct, hash_table):
    """
    Checks if a move is being stored in the transposition table for the given board struct, and if there is, that the
    move is legal.  If it does find a legal move, it stores the move in the struct's first move index, sets the
    struct's next move score to a constant value, and sets it's children_left to a specified constant to indicate there
    was a legal move found in the tt.

    :return: True if a move is found, or False if not.
    """
    node_entry = hash_table[board_struct['hash'] & TT_HASH_MASK]
    if node_entry['depth'] != NO_TT_ENTRY_VALUE:
        if node_entry['entry_hash'] == board_struct['hash']:
            if node_entry['stored_move'][0] != NO_TT_MOVE_VALUE:
                if is_legal_move(board_struct, node_entry['stored_move']):
                    board_struct['unexplored_moves'][0] = node_entry['stored_move']
                    board_struct['next_move_index'] = 0
                    board_struct['children_left'] = NEXT_MOVE_IS_FROM_TT_VAL
                    return True
    return False


@njit
def child_termination_check_and_move_gen(struct_array, hash_table, node_holder, previous_board_map):
    """
    Things being checked:
    1) Draw by the 50-move rule
    2) Draw by insufficient material
    3) Draw by stalemate
    4) Win/loss by checkmate
    5) Termination by information contained in the TT
    6) Draw by threefold repetition
    """
    for j in range(len(struct_array)):
        if struct_array[j]['depth'] != 0:
            if struct_array[j]["halfmove_clock"] >= 50 or has_insufficient_material(struct_array[j]):
                struct_array[j]['best_value'] = TIE_RESULT_SCORE
                struct_array[j]['terminated'] = True
            elif can_draw_from_repetition(struct_array[j], node_holder.held_node, previous_board_map):
                # This is just assigning a draw value, though it doesn't necessarily imply a draw,
                # just that one can be claimed.   Not sure if this needs to be handled, and if yes how to handle it
                struct_array[j]['terminated'] = True
                struct_array[j]['best_value'] = TIE_RESULT_SCORE
            elif should_terminate_from_tt(struct_array[j], hash_table):
                struct_array[j]['terminated'] = True
            elif has_legal_tt_move(struct_array[j], hash_table):
                pass
            else:
                set_up_move_array(struct_array[j])

        node_holder = node_holder.next_holder


@njit
def create_child_structs(struct_array):
    #This should not copy the entire array, instead only the fields which are not directly written over
    #Also this should not be creating full structs for depth zero nodes, a new dtype will likely need to be created
    #which has a subset of the current ones fields.
    child_array = struct_array.copy()

    new_next_move_values = np.empty_like(struct_array['best_value'])

    #This should be removed, and struct_array['prev_move'] should be used instead.  Numba has been very stuborn in resisting this change
    moves_to_push = np.empty((len(struct_array), 3), dtype=np.uint8)

    for j in range(len(struct_array)):
        child_array[j]['unexplored_moves'][:] = 255
        child_array[j]['unexplored_move_scores'][:] = MIN_FLOAT32_VAL
        child_array[j]['prev_move'][:] = struct_array[j]['unexplored_moves'][struct_array[j]['next_move_index']]
        child_array[j]['depth'] = struct_array[j]['depth'] - 1

        moves_to_push[j] = struct_array[j]['unexplored_moves'][struct_array[j]['next_move_index']]

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
    """
    for j in range(len(struct_array)):
        if to_check_mask[j]:
            struct_array[j]['children_left'] = 0
            set_up_move_array_except_move(struct_array[j], struct_array[j]['unexplored_moves'][0].copy())
            struct_array[j]['children_left'] += 1


@njit
def create_holder_for_structs(struct_array, parent_holder, to_create_mask, starting_holder=None):
    found = 0
    for j in range(len(struct_array)):
        if to_create_mask[j]:
            starting_holder = GameNodeHolder(GameNode(struct_array[j:j + 1], parent_holder.held_node), starting_holder)
            found += 1
        parent_holder = parent_holder.next_holder

    return starting_holder, found


@njit
def create_new_holders_and_filter_old(root, node_linked_list, have_children_left_mask, child_struct, create_holder_mask):
    """
    Creates new GameNodes and GameNodeHolders for the given new children, and filters the parents which don't need
    to be given to the open node holder for re-insertion.  Their node holder are connected, and appended to the given root.
    """
    new_child_nodes, num_new_children = create_holder_for_structs(child_struct, node_linked_list, create_holder_mask)

    num_not_filtered = filter_holders_then_append(root, node_linked_list, have_children_left_mask, new_child_nodes)

    return num_new_children, num_new_children + num_not_filtered

@njit
def get_struct_array_from_node_holder(node_holder, length):
    to_return = np.empty(length, dtype=numpy_node_info_dtype)
    for j in range(length):
        to_return[j] = node_holder.struct
        node_holder = node_holder.next_holder
    return to_return


@njit
def update_node_from_value(node, value, following_move, hash_table, new_termination=True):
    if not node is None:
        if node.struct['terminated']:
            if value > node.struct['best_value']:
                node.struct['best_value'] = value

                tt.add_board_and_move_to_tt(node.struct, following_move, hash_table)
                update_node_from_value(node.parent, - value, node.struct['prev_move'], hash_table, False)
        else:
            if new_termination:
                node.struct['children_left'] -= 1

            node.struct['best_value'] = np.maximum(value, node.struct['best_value'])

            if node.struct['best_value'] >= node.struct['separator'] or node.struct['children_left'] == 0:
                node.struct['terminated'] = True

                tt.add_board_and_move_to_tt(node.struct, following_move, hash_table)

                update_node_from_value(node.parent, - node.struct['best_value'], node.struct['prev_move'], hash_table)


@njit
def update_tree_from_terminating_nodes(parent_node_holder, struct_array, hash_table, was_evaluated_mask, eval_results):
    """
    Updates the search tree from the nodes in the current batch which are terminating, this includes all nodes which
    have been marked terminated, or are depth zero.  It also updates the transposition table as needed.
    """
    should_update_mask = np.logical_or(struct_array['depth'] == 0, struct_array['terminated'])

    if eval_results[0] != MAX_FLOAT32_VAL:  #This would be a None parameter but Numba won't let it compile so this is used instead
        tt.add_evaluated_boards_to_tt(struct_array, was_evaluated_mask, eval_results, hash_table)

        eval_results_for_parents = - eval_results


    index_in_evaluations = 0
    for j in range(len(struct_array)):
        if was_evaluated_mask[j]:
            update_node_from_value(
                parent_node_holder.held_node,
                eval_results_for_parents[index_in_evaluations],
                struct_array[j]['prev_move'],
                hash_table)
            index_in_evaluations += 1

        elif should_update_mask[j]:
            update_node_from_value(
                parent_node_holder.held_node,
                - struct_array[j]['best_value'],
                struct_array[j]['prev_move'],
                hash_table)

        parent_node_holder = parent_node_holder.next_holder


@njit
def set_nodes_to_altered_structs(node_holder, struct_array, to_do_mask):
    for j in range(len(struct_array)):
        if to_do_mask[j]:
            node_holder.struct['unexplored_moves'][:] = struct_array[j]['unexplored_moves'][:]
            node_holder.struct['unexplored_move_scores'][:] = struct_array[j]['unexplored_move_scores'][:]

            node_holder.struct['children_left'] = struct_array[j]['children_left']
            node_holder.struct['next_move_index'] = struct_array[j]['next_move_index']

        node_holder = node_holder.next_holder


@njit
def set_child_move_scores(child_structs, scored_child_mask, scores, score_size_array, cum_sum_sizes):
    next_move_scores = np.empty(len(score_size_array),dtype=np.float32)
    num_completed = 0
    for j in range(len(child_structs)):
        if scored_child_mask[j]:
            cur_score_size = score_size_array[num_completed]
            cur_cum_sum_size = cum_sum_sizes[num_completed]

            child_structs[j]['unexplored_move_scores'][:cur_score_size] = scores[cur_cum_sum_size - cur_score_size:cur_cum_sum_size]

            next_move_scores[num_completed] = set_up_next_best_move(child_structs[j])
            num_completed += 1
    return next_move_scores


@njit
def get_move_from_and_filter_squares_and_sizes(child_structs, not_child_structs, child_mask, not_child_mask, num_scored, num_children):
    size_array = np.empty(num_scored, np.uint8)
    total_num_children = len(child_structs)

    size_array[:num_children] = child_structs['children_left'][child_mask]
    size_array[num_children:] = not_child_structs['children_left'][not_child_mask] - 1  #subtracting 1 here to account for the TT move which doesn't need to be scored

    move_indices = np.empty((np.sum(size_array), 2), dtype=np.uint8)

    cur_start_index = 0
    scored_so_far = 0
    for j in range(len(child_structs) + len(not_child_structs)):
        if j < len(child_structs):
            was_scored = child_mask[j]
            struct = child_structs[j]
        else:
            was_scored = not_child_mask[j - total_num_children]
            struct = not_child_structs[j - total_num_children]

        if was_scored:
            cur_size = size_array[scored_so_far]

            if struct['turn']:
                relevant_moves = struct['unexplored_moves'][:cur_size,:2]
            else:
                relevant_moves = SQUARES_180[struct['unexplored_moves'][:cur_size,:2].ravel()].reshape((-1, 2))

            move_indices[cur_start_index:cur_start_index + cur_size, 0] = relevant_moves[:, 0]

            #The following loop is needed since Numba can't handle more than 1 advanced index
            for i in range(cur_size):
                move_indices[cur_start_index + i, 1] = MOVE_FILTER_LOOKUP[relevant_moves[i, 0], relevant_moves[i, 1], struct['unexplored_moves'][i, 2]]

            cur_start_index += cur_size
            scored_so_far += 1

    return size_array, move_indices


@njit
def prepare_to_finish_move_scoring(child_structs, adult_structs, scored_child_mask, scored_adult_mask,
                                   num_scored_children, num_scored_adults):
    size_array, from_to_squares = get_move_from_and_filter_squares_and_sizes(
        child_structs,
        adult_structs,
        scored_child_mask,
        scored_adult_mask,
        num_scored_children + num_scored_adults,
        num_scored_children)

    cum_sum_sizes = np.cumsum(size_array)

    return size_array, from_to_squares, cum_sum_sizes


@njit
def complete_move_evaluation(scores, child_structs, adult_nodes, scored_child_mask, scored_adult_mask,
                             num_children, size_array, cum_sum_sizes):
    adult_next_move_scores = np.empty(len(size_array) - num_children, dtype=np.float32)

    child_next_move_scores = set_child_move_scores(
        child_structs,
        scored_child_mask,
        scores,
        size_array[:num_children],
        cum_sum_sizes[:num_children])

    cur_adult_index = 0
    for j in range(len(child_structs)):
        if scored_adult_mask[j]:
            cur_index = cur_adult_index + num_children
            cur_size = size_array[cur_index]
            cur_cum_sum_size = cum_sum_sizes[cur_index]
            cur_node = adult_nodes.held_node

            cur_node.struct['unexplored_move_scores'][:cur_size] = scores[
                                                                            cur_cum_sum_size - cur_size: cur_cum_sum_size]
            adult_next_move_scores[cur_adult_index] = set_up_next_best_move(cur_node.struct)

            cur_adult_index += 1

        adult_nodes = adult_nodes.next_holder

    return child_next_move_scores, adult_next_move_scores


def do_iteration(node_linked_list, hash_table, previous_board_map, board_eval_fn, move_eval_fn):
    length_of_batch = len_node_holder(node_linked_list)  #this can and should be given to this function
    struct_batch = get_struct_array_from_node_holder(node_linked_list, length_of_batch)

    child_struct, struct_batch_next_move_scores = create_child_structs(struct_batch)

    child_was_from_tt_move_mask = struct_batch['children_left'] == NEXT_MOVE_IS_FROM_TT_VAL

    depth_zero_children_mask = child_struct['depth'] == 0
    depth_not_zero_mask = np.logical_not(depth_zero_children_mask)

    depth_zero_should_terminate_array(child_struct, hash_table, previous_board_map, node_linked_list)

    depth_zero_not_scored_mask = np.logical_and(depth_zero_children_mask, np.logical_not(child_struct['terminated']))

    if np.any(depth_zero_not_scored_mask):
        evaluation_thread, evaluation_scores = start_board_evaluations(
            child_struct,
            depth_zero_not_scored_mask,
            board_eval_fn)
    else:
        evaluation_thread = None
        evaluation_scores = None


    generate_moves_for_tt_move_nodes(struct_batch, child_was_from_tt_move_mask)

    not_one_child_left_mask = struct_batch['children_left'] != 1

    not_only_move_was_tt_move_mask = np.logical_or(not_one_child_left_mask, np.logical_not(child_was_from_tt_move_mask))
    tt_move_nodes_with_more_kids_mask = np.logical_and(not_one_child_left_mask, child_was_from_tt_move_mask)

    child_termination_check_and_move_gen(child_struct, hash_table, node_linked_list, previous_board_map)

    non_zerod_child_not_term_mask = np.logical_and(
        depth_not_zero_mask,
        np.logical_not(child_struct['terminated']))

    non_zerod_kids_for_move_scoring_mask = np.logical_and(
        non_zerod_child_not_term_mask,
        child_struct['children_left'] != NEXT_MOVE_IS_FROM_TT_VAL)

    # Now that staging has been implemented for move scoring, this must be started as soon as it knows exactly which
    # nodes have moves to be scored.  This likely involves stopping the move generation when the first move for each
    # board is discovered, and resuming after the boards which have moves to score have been given to TensorFlow
    # (or after a thread with that task has been started)
    if np.any(non_zerod_kids_for_move_scoring_mask) or np.any(tt_move_nodes_with_more_kids_mask):
        move_thread, move_score_getter, num_children_move_scoring, num_adult_move_scoring = start_move_scoring(
            child_struct,
            struct_batch,
            non_zerod_kids_for_move_scoring_mask,
            tt_move_nodes_with_more_kids_mask,
            move_eval_fn)
    else:
        move_thread = None
        child_next_move_scores = None
        not_child_next_move_scores = None

    # A mask of the given batch which have more unexplored children left
    have_children_left_mask = np.logical_and(
        struct_batch["next_move_index"] != NO_MORE_MOVES_VALUE,
        not_only_move_was_tt_move_mask)


    set_nodes_to_altered_structs(
        node_linked_list,
        struct_batch,
        have_children_left_mask)

    if not evaluation_thread is None:
        evaluation_thread.join()

    if not move_thread is None:
        move_completion_info = prepare_to_finish_move_scoring(
            child_struct,
            struct_batch,
            non_zerod_kids_for_move_scoring_mask,
            tt_move_nodes_with_more_kids_mask,
            num_children_move_scoring,
            num_adult_move_scoring)

    update_tree_from_terminating_nodes(
        node_linked_list,
        child_struct,
        hash_table,
        depth_zero_not_scored_mask,
        evaluation_scores if not evaluation_scores is None else INT_ARRAY_NONE)

    if not move_thread is None:
        move_thread.join()

        move_scores = move_score_getter[0](
            [move_completion_info[1][:, 0],
             move_completion_info[1][:, 1],
             move_completion_info[0]])

        child_next_move_scores, not_child_next_move_scores = complete_move_evaluation(
            scores=move_scores,
            child_structs=child_struct,
            adult_nodes=node_linked_list,
            scored_child_mask=non_zerod_kids_for_move_scoring_mask,
            scored_adult_mask=tt_move_nodes_with_more_kids_mask,
            num_children=num_children_move_scoring,
            size_array=move_completion_info[0],
            cum_sum_sizes=move_completion_info[2])

    dummy_root = create_dummy_node_holder()
    num_new_children, num_returning = create_new_holders_and_filter_old(dummy_root, node_linked_list, have_children_left_mask, child_struct, non_zerod_child_not_term_mask)
    to_return = dummy_root.next_holder

    #Set up the array of scores used to place the returned nodes into their proper bins
    scores_to_return = np.full(num_returning, TT_MOVE_SCORE_VALUE, dtype=np.float32)
    num_not_child_scores = num_returning - num_new_children
    if num_not_child_scores != 0:
        scores_to_return[:num_not_child_scores] = struct_batch_next_move_scores[have_children_left_mask]
        if not not_child_next_move_scores is None and len(not_child_next_move_scores) != 0:
            scores_to_return[:num_not_child_scores][scores_to_return[:num_not_child_scores] == TT_MOVE_SCORE_VALUE] = not_child_next_move_scores
    if not child_next_move_scores is None and len(child_next_move_scores) != 0:
            scores_to_return[num_not_child_scores:][child_struct[non_zerod_child_not_term_mask]['children_left'] != NEXT_MOVE_IS_FROM_TT_VAL] = child_next_move_scores


    return to_return, scores_to_return


def zero_window_negamax_search(root_game_node, open_node_holder, board_eval_fn, move_eval_fn, hash_table,
                               previous_board_map):
    next_batch = GameNodeHolder(root_game_node, None)
    while next_batch:
        to_insert, to_insert_scores = do_iteration(
            next_batch, hash_table, previous_board_map, board_eval_fn, move_eval_fn)

        if root_game_node.struct['terminated']:
            open_node_holder.clear_list()
            break

        if not to_insert and open_node_holder.is_empty():
            break

        next_batch = open_node_holder.insert_nodes_and_get_next_batch(to_insert, to_insert_scores)

    return root_game_node.struct['best_value']


def set_up_root_node_for_struct(move_eval_fn, hash_table, previous_board_map, root_struct):
    if not root_struct['turn']:
        root_struct = convert_board_to_whites_perspective(root_struct)

    root_node = GameNode(root_struct, None)

    temp_game_node_holder = GameNodeHolder(root_node, None)

    struct_array = root_node.board_struct

    child_termination_check_and_move_gen(
        struct_array,
        hash_table,
        temp_game_node_holder,
        previous_board_map)

    num_moves_to_score = struct_array[0]['children_left']
    num_moves_to_score_as_array = np.array([num_moves_to_score])

    if struct_array[0]['terminated'] or num_moves_to_score == NEXT_MOVE_IS_FROM_TT_VAL:
        return root_node

    move_thread, move_score_getter, _, _ = start_move_scoring(
        struct_array,
        struct_array,
        np.zeros(1, dtype=np.bool_),
        np.ones(1, dtype=np.bool_),
        move_eval_fn)


    move_thread.join()

    relevant_moves = struct_array[0]['unexplored_moves'][:num_moves_to_score]

    if not struct_array[0]['turn']:
        relevant_moves[:, :2] = SQUARES_180[relevant_moves[:, :2]]

    move_filters = MOVE_FILTER_LOOKUP[relevant_moves[:, 0], relevant_moves[:, 1], relevant_moves[:, 2]]
    move_from_squares = relevant_moves[:, 0]

    scores = move_score_getter[0]([move_from_squares, move_filters, num_moves_to_score_as_array])

    complete_move_evaluation(
        scores,
        struct_array,
        temp_game_node_holder,
        np.zeros(1, dtype=np.bool_),
        np.ones(1, dtype=np.bool_),
        0,
        num_moves_to_score_as_array,
        num_moves_to_score_as_array)

    return root_node


def set_up_root_node_from_fen(move_eval_fn, hash_table, previous_board_map, fen, depth=255, separator=0):
    return set_up_root_node_for_struct(
        move_eval_fn,
        hash_table,
        previous_board_map,
        create_node_info_from_fen(fen, depth, separator))


def mtd_f(fen, depth, first_guess, open_node_holder, board_eval_fn, move_eval_fn, hash_table, previous_board_map,
          guess_increment=.05, print_info=False):
    """
    Does an mtd(f) search modified to a binary search (this is done to address the granularity of the evaluation network).
    """
    cur_guess = first_guess

    upper_bound = WIN_RESULT_SCORES[0]
    lower_bound = LOSS_RESULT_SCORES[0]

    if print_info:
        counter = 0

    while lower_bound < upper_bound:

        if lower_bound == LOSS_RESULT_SCORES[0]:
            if upper_bound != WIN_RESULT_SCORES[0]:
                beta = np.minimum(upper_bound - guess_increment, np.nextafter(upper_bound, MIN_FLOAT32_VAL))
            else:
                beta = cur_guess
        elif upper_bound == WIN_RESULT_SCORES[0]:
            beta = np.maximum(lower_bound + guess_increment, np.nextafter(lower_bound, MAX_FLOAT32_VAL))
        else:
            beta = np.maximum(lower_bound + (upper_bound - lower_bound) / 2, np.nextafter(lower_bound, MAX_FLOAT32_VAL))

        seperator_to_use = np.nextafter(beta, MIN_FLOAT32_VAL)

        # This would ideally share the same tree, but updated for the new separation value
        cur_root_node = set_up_root_node_from_fen(move_eval_fn, hash_table, previous_board_map, fen, depth, seperator_to_use)

        if cur_root_node.struct['terminated']:
            cur_guess = cur_root_node.struct['best_value']
        else:
            cur_guess = zero_window_negamax_search(
                cur_root_node,
                open_node_holder,
                board_eval_fn,
                move_eval_fn,
                hash_table=hash_table,
                previous_board_map=previous_board_map)

        if cur_guess < beta:
            upper_bound = cur_guess
        else:
            lower_bound = cur_guess

        if print_info:
            counter += 1
            print("Finished iteration %d with lower and upper bounds (%f,%f) after search returned %f" % (counter, lower_bound, upper_bound, cur_guess))

    tt_move = tt.choose_move(hash_table, cur_root_node, fen.split()[1]=='b')

    return cur_guess, tt_move, hash_table


def iterative_deepening_mtd_f(fen, depths_to_search, open_node_holder, board_eval_fn, move_eval_fn, hash_table,
                              previous_board_map, first_guess=0, guess_increments=None, print_info=False):
    if guess_increments is None:
        guess_increments = [.05]*len(depths_to_search)

    if print_info:
        start_time = time.time()


    for depth, increment in zip(depths_to_search, guess_increments):
        if print_info:
            print("Starting depth %d search, with first guess %f"%(depth, first_guess))
            started_search = time.time()

        first_guess, tt_move, hash_table = mtd_f(
            fen,
            depth,
            first_guess,
            open_node_holder,
            board_eval_fn,
            move_eval_fn,
            hash_table=hash_table,
            previous_board_map=previous_board_map,
            guess_increment=increment,
            print_info=print_info)


        if print_info:
            print("Completed depth %d in time %f with value %f\n"%(depth, time.time() - started_search, first_guess))


    if print_info:
        print("The nodes processed per second (including repeats) is:", open_node_holder.total_out/(time.time() - start_time))
        print("The number of nodes inserted into the node list (including repeats) was %d, and %d were retrieved.\n" % (open_node_holder.total_in, open_node_holder.total_out))
        open_node_holder.reset_logs()


    return first_guess, tt_move, hash_table
