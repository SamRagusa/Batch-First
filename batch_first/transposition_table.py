import numpy as np
import numba as nb

import chess


from batch_first import SIZE_EXPONENT_OF_TWO_FOR_TT_INDICES, ONES_IN_RELEVANT_BITS_FOR_TT_INDEX, NO_TT_ENTRY_VALUE,\
    NO_TT_MOVE_VALUE, MIN_FLOAT32_VAL, MAX_FLOAT32_VAL






hash_table_numpy_dtype = np.dtype([("entry_hash", np.uint64),
                                   ("depth", np.uint8),
                                   ("upper_bound", np.float32),
                                   ("lower_bound", np.float32),
                                   ("stored_from_square", np.uint8),
                                   ("stored_to_square", np.uint8),
                                   ("stored_promotion", np.uint8)])

hash_table_numba_dtype = nb.from_dtype(hash_table_numpy_dtype)



def get_empty_hash_table():
    """
    NOTES:
    1) Uses global variable SIZE_EXPONENT_OF_TWO_FOR_TT_INDICES for it's size.
    """
    return np.array(
        [(np.uint64(0),
          NO_TT_ENTRY_VALUE,
          MAX_FLOAT32_VAL,
          MIN_FLOAT32_VAL,
          NO_TT_MOVE_VALUE,
          NO_TT_MOVE_VALUE,
          NO_TT_MOVE_VALUE) for _ in range(2 ** SIZE_EXPONENT_OF_TWO_FOR_TT_INDICES)],
        dtype=hash_table_numpy_dtype)



def choose_move(hash_table, node):
    """
    Chooses the desired move to be made from the given node.  This is done by use of the given hash table.

    NOTES:
    1) This is a first attempt, and while EXTREMELY crucial, I don't believe it works as desired.

    :return: A python-chess Move object representing the desired move to be made
    """
    root_tt_entry = hash_table[np.uint64(node.board_struct[0]['hash']) & ONES_IN_RELEVANT_BITS_FOR_TT_INDEX]
    return chess.Move(
        np.int8(root_tt_entry["stored_from_square"]),
        np.int8(root_tt_entry["stored_to_square"]),
        np.int8(root_tt_entry["stored_promotion"]) if root_tt_entry["stored_promotion"]!=0 else None)



@nb.njit
def set_tt_node(hash_table, board_hash, depth, upper_bound=MAX_FLOAT32_VAL, lower_bound=MIN_FLOAT32_VAL):
    """
    Puts the given information about a node into the hash table.
    """
    index = board_hash & ONES_IN_RELEVANT_BITS_FOR_TT_INDEX  #THIS IS BEING DONE TWICE
    hash_table[index]["entry_hash"] = board_hash
    hash_table[index]["depth"] = depth
    hash_table[index]["upper_bound"] = upper_bound
    hash_table[index]["lower_bound"] = lower_bound


@nb.njit
def set_tt_move(hash_table, tt_index, following_move):
    """
    Write the given move in the given hash table, at the given index.
    """
    hash_table[tt_index]["stored_from_square"] = following_move[0]
    hash_table[tt_index]["stored_to_square"] = following_move[1]
    hash_table[tt_index]["stored_promotion"] = following_move[2]


@nb.njit
def wipe_tt_move(hash_table, tt_index):
    """
    Writes over the move in the given hash_table at the given index.  It sets all the move values to NO_TT_MOVE_VALUE.
    """
    hash_table[tt_index]["stored_from_square"] = NO_TT_MOVE_VALUE
    hash_table[tt_index]["stored_to_square"] = NO_TT_MOVE_VALUE
    hash_table[tt_index]["stored_promotion"] = NO_TT_MOVE_VALUE


@nb.njit
def add_board_and_move_to_tt(board_struct, following_move, hash_table):
    """
    Adds the information about a current board and the move which was made previously, to the
    transposition table.

    NOTES:
    1) While this currently does work, it represents one of the most crucial components of the negamax search,
    and has not been given the thought and effort it needs.  This is an extremely high priority
    """
    tt_index = board_struct['hash'] & ONES_IN_RELEVANT_BITS_FOR_TT_INDEX
    node_entry = hash_table[tt_index]
    if node_entry["depth"] != NO_TT_ENTRY_VALUE:
        if node_entry["entry_hash"] == board_struct['hash']:
            if node_entry["depth"] == board_struct['depth']:
                if board_struct['best_value'] >= board_struct['separator']:
                    if board_struct['best_value'] > node_entry["lower_bound"]:
                        node_entry["lower_bound"] = board_struct['best_value']
                        set_tt_move(hash_table, tt_index, following_move)

                elif board_struct['best_value'] < node_entry["upper_bound"]:
                    node_entry["upper_bound"] = board_struct['best_value']
            elif node_entry["depth"] < board_struct['depth']:
                # Overwrite the data currently stored in the hash table
                if board_struct['best_value'] >= board_struct['separator']:
                    set_tt_node(hash_table, board_struct['hash'], board_struct['depth'], lower_bound=board_struct['best_value'])  ###########THIS IS WRITING THE HASH EVEN THOUGH IT HAS NOT CHANGED########
                    set_tt_move(hash_table, tt_index, following_move)
                else:
                    set_tt_node(hash_table, board_struct['hash'], board_struct['depth'],upper_bound=board_struct['best_value'])

            # Don't change anything if it's depth is less than the depth in the TT
        else:
            # Using the always replace scheme for simplicity and easy implementation (likely only for now)
            if board_struct['best_value'] >= board_struct['separator']:
                set_tt_node(hash_table, board_struct['hash'], board_struct['depth'],lower_bound=board_struct['best_value'])
                set_tt_move(hash_table, tt_index, following_move)
            else:
                set_tt_node(hash_table, board_struct['hash'], board_struct['depth'],upper_bound=board_struct['best_value'])
                wipe_tt_move(hash_table, tt_index)
    else:

        if board_struct['best_value'] >= board_struct['separator']:
            set_tt_node(hash_table, board_struct['hash'], board_struct['depth'],lower_bound=board_struct['best_value'])
            set_tt_move(hash_table, tt_index, following_move)
        else:
            set_tt_node(hash_table, board_struct['hash'], board_struct['depth'],upper_bound=board_struct['best_value'])




@nb.njit
def add_boards_to_tt(struct_array, hash_table):
    """
    IMPORTANT NOTES:
    1) This function has been commented out because it currently isn't correct (the logic, not the implementation).
    The issue is the TT doesn't know how to store depth zero nodes yet, this is a very high priority and will be
    implemented properly as soon as possible.
    """
    return
    # for j in range(len(struct_array)):
    #
    #     node_entry = hash_table[struct_array[j]['hash'] & ONES_IN_RELEVANT_BITS]
    #     if node_entry["depth"] != NO_TT_ENTRY_VALUE:
    #         if node_entry["entry_hash"] == struct_array[j]['hash']:
    #             if node_entry["depth"] == struct_array[j]['depth']:
    #                 if struct_array[j]['best_value'] >= struct_array[j]['separator']:
    #                     if struct_array[j]['best_value'] > node_entry["lower_bound"]:
    #                         node_entry["lower_bound"] = struct_array[j]['best_value']
    #                 elif struct_array[j]['best_value'] < node_entry["upper_bound"]:
    #                     node_entry["upper_bound"] = struct_array[j]['best_value']
    #             elif node_entry["depth"] < struct_array[j]['depth']:
    #                 #Overwrite the data currently stored in the hash table
    #                 if struct_array[j]['best_value'] >= struct_array[j]['separator']:
    #                     set_tt_node(hash_table, struct_array[j]['hash'], struct_array[j]['depth'],upper_bound=MAX_FLOAT32_VAL, lower_bound=struct_array[j]['best_value'])   ###########THIS IS WRITING THE HASH EVEN THOUGH IT HAS NOT CHANGED########
    #                 # else:
    #                 #     set_tt_node(hash_table, struct_array[j]['hash'], struct_array[j]['depth'],upper_bound=game_node.best_value)
    #
    #             # Don't change anything if it's depth is less than the depth in the TT
    #         else:
    #             #Using the always replace scheme for simplicity and easy implementation (likely only for now)
    #             # print("A hash table entry exists with a different hash than wanting to be inserted!")
    #             if struct_array[j]['best_value']>= struct_array[j]['separator']:
    #                 set_tt_node(hash_table, struct_array[j]['hash'], struct_array[j]['depth'], lower_bound=struct_array[j]['best_value'])
    #             else:
    #                 set_tt_node(hash_table, struct_array[j]['hash'], struct_array[j]['depth'], upper_bound=struct_array[j]['best_value'])
    #     else:
    #
    #         if struct_array[j]['best_value'] >= struct_array[j]['separator']:
    #             set_tt_node(hash_table, struct_array[j]['hash'], struct_array[j]['depth'], lower_bound=struct_array[j]['best_value'])
    #         else:
    #             set_tt_node(hash_table, struct_array[j]['hash'], struct_array[j]['depth'], upper_bound=struct_array[j]['best_value'])

