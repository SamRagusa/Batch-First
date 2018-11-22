from . import *

from .numba_board import square_mirror



hash_table_numpy_dtype = np.dtype([("entry_hash", np.uint64),  #Total of 160 bits per entry
                                   ("depth", np.uint8),
                                   ("upper_bound", np.float32),
                                   ("lower_bound", np.float32),
                                   ("stored_move", np.uint8, (3))])

hash_table_numba_dtype = nb.from_dtype(hash_table_numpy_dtype)


blank_tt_entry = np.array([(
        0,
        NO_TT_ENTRY_VALUE,
        MAX_FLOAT32_VAL,
        MIN_FLOAT32_VAL,
        np.full(3, NO_TT_MOVE_VALUE, dtype=np.uint8))], dtype=hash_table_numpy_dtype)[0]



def get_empty_hash_table():
    return np.full(2**SIZE_EXPONENT_OF_TWO_FOR_TT_INDICES, blank_tt_entry)


@njit
def clear_hash_table(table):
    table[table['depth'] != NO_TT_ENTRY_VALUE] = blank_tt_entry
    return table


def choose_move(hash_table, node, flip_move=False):
    """
    Chooses the desired move to be made from the given node.  This is done by use of the given hash table.

    :return: A python-chess Move object representing the desired move to be made
    """
    root_tt_entry = hash_table[np.uint64(node.struct['hash']) & TT_HASH_MASK]
    move_array = root_tt_entry['stored_move']

    if flip_move:
        move_array[:-1] = square_mirror(move_array[:-1])

    return chess.Move(
        move_array[0].view(np.int8),
        move_array[1].view(np.int8),
        None if move_array[2]==0 else move_array[2].view(np.int8))


@nb.njit
def set_tt_node(hash_entry, board_hash, depth, overwrite_hash=True, overwrite_bounds=False,
                upper_bound=MAX_FLOAT32_VAL, lower_bound=MIN_FLOAT32_VAL):
    """
    Puts the given information about a node into the hash table.
    """
    hash_entry['depth'] = depth

    if overwrite_hash:
        hash_entry['entry_hash'] = board_hash

    if upper_bound != MAX_FLOAT32_VAL or overwrite_bounds:
        hash_entry['upper_bound'] = upper_bound
    if lower_bound != MIN_FLOAT32_VAL or overwrite_bounds:
        hash_entry['lower_bound'] = lower_bound


@nb.njit
def set_tt_move(hash_entry, following_move):
    """
    Write the given move in the given hash table, at the given index.
    """
    hash_entry['stored_move'][:] = following_move


@nb.njit
def wipe_tt_move(hash_entry):
    """
    Writes over the move in the given hash_table at the given index.  It sets all the move values to NO_TT_MOVE_VALUE.
    """
    hash_entry['stored_move'][:] = NO_TT_MOVE_VALUE


@nb.njit
def add_board_and_move_to_tt(board_struct, following_move, hash_table):
    """
    Adds the information about a current board and the move which was made previously, to the
    transposition table.

    NOTES:
    1) While this currently does work, it represents one of the most crucial components of the negamax search,
    and has not been given the thought and effort it needs.  This is an extremely high priority
    """
    node_entry = hash_table[board_struct['hash'] & TT_HASH_MASK]
    if node_entry['depth'] != NO_TT_ENTRY_VALUE:
        if node_entry['entry_hash'] == board_struct['hash']:
            if node_entry['depth'] == board_struct['depth']:
                if board_struct['best_value'] >= board_struct['separator']:
                    if board_struct['best_value'] > node_entry['lower_bound']:
                        node_entry['lower_bound'] = board_struct['best_value']
                        set_tt_move(node_entry, following_move)

                elif board_struct['best_value'] < node_entry['upper_bound']:
                    node_entry['upper_bound'] = board_struct['best_value']

            elif node_entry['depth'] < board_struct['depth']:
                # Overwrite the data currently stored in the hash table
                if board_struct['best_value'] >= board_struct['separator']:
                    set_tt_move(node_entry, following_move)
                    set_tt_node(node_entry, board_struct['hash'], board_struct['depth'],
                                lower_bound=board_struct['best_value'], overwrite_hash=False, overwrite_bounds=True)
                else:
                    set_tt_node(node_entry, board_struct['hash'], board_struct['depth'],
                                upper_bound=board_struct['best_value'], overwrite_hash=False, overwrite_bounds=True)
            # Don't change anything if it's depth is less than the depth in the TT
        else:
            # Using the always replace scheme for simplicity and easy implementation (likely only for now)
            if board_struct['best_value'] >= board_struct['separator']:
                set_tt_move(node_entry, following_move)
                set_tt_node(node_entry, board_struct['hash'], board_struct['depth'],
                            lower_bound=board_struct['best_value'], overwrite_bounds=True)
            else:
                wipe_tt_move(node_entry)
                set_tt_node(node_entry, board_struct['hash'], board_struct['depth'],
                            upper_bound=board_struct['best_value'], overwrite_bounds=True)
    else:
        if board_struct['best_value'] >= board_struct['separator']:
            set_tt_move(node_entry, following_move)
            set_tt_node(node_entry, board_struct['hash'], board_struct['depth'], lower_bound=board_struct['best_value'])
        else:
            set_tt_node(node_entry, board_struct['hash'], board_struct['depth'], upper_bound=board_struct['best_value'])




@nb.njit
def add_evaluated_boards_to_tt(struct_array, was_evaluated_mask, eval_results, hash_table):
    num_done = 0
    for j in range(len(struct_array)):
        if was_evaluated_mask[j]:
            node_entry = hash_table[struct_array[j]['hash'] & TT_HASH_MASK]
            if node_entry['depth'] == NO_TT_ENTRY_VALUE:
                cur_result = eval_results[num_done]
                set_tt_node(node_entry, struct_array[j]['hash'], 0, lower_bound=cur_result, upper_bound=cur_result)

            num_done += 1