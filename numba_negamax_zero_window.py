import numba as nb
import threading
from concurrent.futures import ThreadPoolExecutor
from chess import uci
import random as rand

from numba import float32

from numba_board import *

# from temp_board_eval_client import ANNEvaluator
from global_open_priority_nodes import PriorityBins

from evaluation_ann import main as evaluation_ann_main
from move_evaluation_ann import main as move_ann_main

import numpy as np

if __name__ == "__main__":
    PREDICTOR, CLOSER = evaluation_ann_main([True])
    MOVE_PREDICTOR, MOVE_CLOSER = move_ann_main([True])




MIN_FLOAT32_VAL = np.finfo(np.float32).min
MAX_FLOAT32_VAL = np.finfo(np.float32).max
FLOAT32_EPS = np.finfo(np.float32).eps
ALMOST_MIN_FLOAT_32_VAL = MIN_FLOAT32_VAL + FLOAT32_EPS
ALMOST_MAX_FLOAT_32_VAL = MAX_FLOAT32_VAL - FLOAT32_EPS

WIN_RESULT_SCORE = ALMOST_MAX_FLOAT_32_VAL
LOSS_RESULT_SCORE = ALMOST_MIN_FLOAT_32_VAL
TIE_RESULT_SCORE = np.float32(0)

#This value is used for indicating that a given node has/had a legal move in the transposition table that it will/would
#expand prior to it's full move generation and scoring.
NEXT_MOVE_IS_FROM_TT_VAL = np.uint8(254)

#This value is used for indicating that a move in a transposition table entry is not being stored.
NO_TT_MOVE_VALUE = np.uint8(255)

#This value is used for indicating that no entry in the transposition table exists for that hash.  It is stored as the
#entries depth.
NO_TT_ENTRY_VALUE = np.uint8(255)

#This value is the value used to assigned a node who's next move was found in the TT to the desired bin.
TT_MOVE_SCORE_VALUE = ALMOST_MAX_FLOAT_32_VAL






hash_table_numpy_dtype = np.dtype([("entry_hash", np.uint64),
                                   ("depth", np.uint8),
                                   ("upper_bound", np.float32),
                                   ("lower_bound", np.float32),
                                   ("stored_from_square", np.uint8),
                                   ("stored_to_square", np.uint8),
                                   ("stored_promotion", np.uint8)])

hash_table_numba_dtype = nb.from_dtype(hash_table_numpy_dtype)



SIZE_EXPONENT_OF_TWO = 20 #This needs to be more precicely picked.
ONES_IN_RELEVANT_BITS = np.uint64(2**(SIZE_EXPONENT_OF_TWO)-1)


switch_square_fn = lambda x : 8 * (7 - (x >> 3)) + (x & 7)
MOVE_TO_INDEX_ARRAY = np.zeros(shape=[64,64],dtype=np.int32)
REVERSED_MOVE_TO_INDEX_ARRAY = np.zeros(shape=[64,64],dtype=np.int32)

for key, value in generate_move_to_enumeration_dict().items():
    MOVE_TO_INDEX_ARRAY[key[0],key[1]] = value
    REVERSED_MOVE_TO_INDEX_ARRAY[switch_square_fn(key[0]), switch_square_fn(key[1])] = value

numpy_board_state_dtype = np.dtype([("pawns", np.uint64),
                                    ("knights", np.uint64),
                                    ("bishops", np.uint64),
                                    ("rooks", np.uint64),
                                    ("queens", np.uint64),
                                    ("kings", np.uint64),
                                    ("occupied_w", np.uint64),
                                    ("occupied_b", np.uint64),
                                    ("occupied", np.uint64),
                                    ("turn", np.bool_),
                                    ("castling_rights", np.uint64),
                                    ("ep_square", np.uint8),
                                    ("halfmove_clock", np.uint8),
                                    ("cur_hash", np.uint64)])

numba_numpy_board_type = nb.from_dtype(numpy_board_state_dtype)


@njit
def numpy_scalar_to_jitclass(scalar):
    temp_ep_value = None if scalar["ep_square"] == 255 else scalar["ep_square"]
    return BoardState(scalar["pawns"],
                      scalar["knights"],
                      scalar["bishops"],
                      scalar["rooks"],
                      scalar["queens"],
                      scalar["kings"],
                      scalar["occupied_w"],
                      scalar["occupied_b"],
                      scalar["occupied"],
                      scalar["turn"],
                      scalar["castling_rights"],
                      temp_ep_value,
                      scalar["halfmove_clock"],
                      scalar["cur_hash"])
# @njit
def jitclass_to_numpy_scalar_array(objects):
    return np.array([(object.pawns, object.knights, object.bishops, object.rooks, object.queens, object.kings,
                      object.occupied_w, object.occupied_b, object.occupied,
                      object.turn, object.castling_rights, 255 if object.ep_square is None else object.ep_square,
                      object.halfmove_clock,
                      object.cur_hash) for object in objects], dtype=numpy_board_state_dtype)






gamenode_type = deferred_type()

gamenode_spec = OrderedDict()

gamenode_spec["board_state"] = BoardState.class_type.instance_type#numba_numpy_board_type

gamenode_spec["separator"] = float32
gamenode_spec["depth"] = uint8

gamenode_spec["parent"] = optional(gamenode_type)
gamenode_spec["prev_move"] = optional(Move.class_type.instance_type)

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
    def __init__(self, board_state, parent, prev_move, depth, separator, best_value, unexplored_moves, unexplored_move_scores, next_move_score, next_move, terminated, children_left):
        self.board_state = board_state

        # Eventually this will likely be some sort of list, so that updating multiple parents is possible
        # when handling transpositions.
        self.parent = parent
        self.prev_move = prev_move

        self.separator = separator
        self.depth = depth

        self.best_value = best_value

        self.next_move_score = next_move_score
        self.next_move = next_move

        self.unexplored_moves = unexplored_moves
        self.unexplored_move_scores = unexplored_move_scores

        self.terminated = terminated

        self.children_left = children_left




gamenode_type.define(GameNode.class_type.instance_type)




@njit(GameNode.class_type.instance_type(GameNode.class_type.instance_type, Move.class_type.instance_type))
def zero_window_create_child_from_move(game_node, move):
    return GameNode(
        copy_push(game_node.board_state, move),
        game_node,
        move,
        game_node.depth - 1,
        -game_node.separator,
        MIN_FLOAT32_VAL,
        None,
        None,
        None,
        None,
        False,
        np.uint8(0))


@njit(boolean(GameNode.class_type.instance_type))
def has_uncreated_child(game_node):
    if game_node.next_move_score is None:
        return False
    return True








def create_game_node_from_fen(fen, depth, seperator):
    return GameNode(create_board_state_from_fen(fen),
                    None,
                    None,
                    depth,
                    seperator,
                    MIN_FLOAT32_VAL,
                    None,
                    None,
                    None,
                    None,
                    False,
                    np.uint8(0))


def create_root_game_node(depth, seperator):
    return create_game_node_from_fen(INITIAL_BOARD_FEN, depth, seperator)



def get_empty_hash_table():
    """
    Uses global variable SIZE_EXPONENT_OF_TWO for it's size.
    """

    return np.array(
        [(np.uint64(0),
          NO_TT_ENTRY_VALUE,
          MAX_FLOAT32_VAL,
          MIN_FLOAT32_VAL,
          NO_TT_MOVE_VALUE,
          NO_TT_MOVE_VALUE,
          NO_TT_MOVE_VALUE) for _ in range(2 ** SIZE_EXPONENT_OF_TWO)],
        dtype=hash_table_numpy_dtype)


@njit#(void(hash_table_numba_dtype[:], uint64, uint8, float32, float32))
def set_tt_node(hash_table, board_hash, depth, upper_bound=MAX_FLOAT32_VAL, lower_bound=MIN_FLOAT32_VAL):
    """
    Puts the given information about a node into the hash table.
    """
    index = board_hash & ONES_IN_RELEVANT_BITS  #THIS IS BEING DONE TWICE
    hash_table[index]["entry_hash"] = board_hash
    hash_table[index]["depth"] = depth
    hash_table[index]["upper_bound"] = upper_bound
    hash_table[index]["lower_bound"] = lower_bound


@njit#(void(hash_table_numba_dtype[:], uint64, Move.class_type.instance_type))
def set_tt_move(hash_table, tt_index, following_move):
    hash_table[tt_index]["stored_from_square"] = following_move.from_square
    hash_table[tt_index]["stored_to_square"] = following_move.to_square
    hash_table[tt_index]["stored_promotion"] = following_move.promotion




@njit#((GameNode.class_type.instance_type, optional(hash_table_numba_dtype[:,:])))
def terminated_from_tt(game_node, hash_table):
    """
    Checks if the node should be terminated from the information in contained in the given hash_table.  It also
    updates the values in the given node when applicable.

    NOTES:
    1) This logic has not been heavily thought out, I've been focused on nodes per second as of now.
    """
    hash_entry = hash_table[game_node.board_state.cur_hash & ONES_IN_RELEVANT_BITS]
    if hash_entry["depth"] != NO_TT_ENTRY_VALUE and hash_entry["entry_hash"] == game_node.board_state.cur_hash:
        if hash_entry["depth"] >= game_node.depth:
            if hash_entry["lower_bound"] >= game_node.separator: #This may want to be >
                game_node.best_value = hash_entry["lower_bound"]
                return True
            else:
                if hash_entry["lower_bound"] > game_node.best_value:
                    game_node.best_value = hash_entry["lower_bound"]

                if hash_entry["upper_bound"] < game_node.separator:  #This may want to be <=
                    # game_node.best_value = hash_entry["upper_bound"] #I'm not confident about this
                    game_node.best_value = hash_entry["lower_bound"] #I'm not confident about this
                    return True
    return False



@njit
def add_one_board_to_tt(game_node, hash_table):
    """
    IMPORTANT NOTES:
    1) THIS CURRENTLY CAUSES ISSUES

    TO-DO:
    1) Build/test a better replacement scheme (currently always replacing)
    """
    node_entry = hash_table[game_node.board_state.cur_hash & ONES_IN_RELEVANT_BITS]
    if node_entry["depth"] != NO_TT_ENTRY_VALUE:
        if node_entry["entry_hash"] == game_node.board_state.cur_hash:
            if node_entry["depth"] == game_node.depth:
                if game_node.best_value >= game_node.separator:
                    if game_node.best_value > node_entry["lower_bound"]:
                        node_entry["lower_bound"] = game_node.best_value
                elif game_node.best_value < node_entry["upper_bound"]:
                    node_entry["upper_bound"] = game_node.best_value
            elif node_entry["depth"] < game_node.depth:
                #Overwrite the data currently stored in the hash table
                if game_node.best_value >= game_node.separator:
                    set_tt_node(hash_table, game_node.board_state.cur_hash, game_node.depth,upper_bound=MAX_FLOAT32_VAL, lower_bound=game_node.best_value)   ###########THIS IS WRITING THE HASH EVEN THOUGH IT HAS NOT CHANGED########
                # else:
                #     set_tt_node(hash_table, game_node.board_state.cur_hash, game_node.depth,upper_bound=game_node.best_value)

            # Don't change anything if it's depth is less than the depth in the TT
        else:
            #Using the always replace scheme for simplicity and easy implementation (likely only for now)
            # print("A hash table entry exists with a different hash than wanting to be inserted!")
            if game_node.best_value >= game_node.separator:
                set_tt_node(hash_table, game_node.board_state.cur_hash, game_node.depth, lower_bound=game_node.best_value)
            else:
                set_tt_node(hash_table, game_node.board_state.cur_hash, game_node.depth, upper_bound=game_node.best_value)
    else:

        if game_node.best_value >= game_node.separator:
            set_tt_node(hash_table, game_node.board_state.cur_hash, game_node.depth, lower_bound=game_node.best_value)
        else:
            set_tt_node(hash_table, game_node.board_state.cur_hash, game_node.depth, upper_bound=game_node.best_value)



@njit
def add_board_and_move_to_tt(game_node, following_move, hash_table):
    """
    This function is the same as add_one_board_to_tt except that it also stores a given move (when needed).

    IMPORTANT NOTES:
    1) THIS CURRENTLY CAUSES ISSUES
    2) Though this may be a tiny bit faster to have separate from add_one_board_to_tt, they should be combined.
    The reason it's currently not is because Numba has rejected all my attempts to compile a combined function.
    """
    tt_index = game_node.board_state.cur_hash & ONES_IN_RELEVANT_BITS
    node_entry = hash_table[tt_index]
    if node_entry["depth"] != NO_TT_ENTRY_VALUE:
        if node_entry["entry_hash"] == game_node.board_state.cur_hash:
            if node_entry["depth"] == game_node.depth:
                if game_node.best_value >= game_node.separator:
                    if game_node.best_value > node_entry["lower_bound"]:
                        node_entry["lower_bound"] = game_node.best_value
                        set_tt_move(hash_table, tt_index, following_move)


                elif game_node.best_value < node_entry["upper_bound"]:
                    node_entry["upper_bound"] = game_node.best_value

            elif node_entry["depth"] < game_node.depth:
                #Overwrite the data currently stored in the hash table
                if game_node.best_value >= game_node.separator:
                    set_tt_node(hash_table, game_node.board_state.cur_hash, game_node.depth, lower_bound=game_node.best_value) #####This is writing the hash even though it doesn't need to
                    set_tt_move(hash_table, tt_index, following_move)
                # else:
                #     set_tt_node(hash_table, game_node.board_state.cur_hash, game_node.depth,upper_bound=game_node.best_value)

            #Don't change anything if it's depth is less than the depth in the TT
        else:
            #Using the always replace scheme for simplicity and easy implementation (likely only for now)
            if game_node.best_value >= game_node.separator:
                set_tt_node(hash_table, game_node.board_state.cur_hash, game_node.depth, lower_bound=game_node.best_value)
                set_tt_move(hash_table, tt_index, following_move)
            else:
                set_tt_node(hash_table, game_node.board_state.cur_hash, game_node.depth, upper_bound=game_node.best_value)
    else:
        if game_node.best_value >= game_node.separator:
            set_tt_node(hash_table, game_node.board_state.cur_hash, game_node.depth, lower_bound=game_node.best_value)
            set_tt_move(hash_table, tt_index, following_move)
        else:
            set_tt_node(hash_table, game_node.board_state.cur_hash, game_node.depth, upper_bound=game_node.best_value)



# @jit(boolean(GameNode.class_type.instance_type),nopython=True)  #This worked, but prevented other functions from compiling
@njit
def should_terminate(game_node):
    cur_node = game_node
    while cur_node is not None:
        if cur_node.terminated:
            return True
        cur_node = cur_node.parent
    return False


def zero_window_update_node_from_value(node, value, following_move, hash_table):
    if not node is None:
        if not node.terminated:  #This may be preventing more accurate predictions when two+ nodes which would terminate a single node in one iteration are updated out of order
            node.children_left -= 1
            # node.children_left = np.uint8(np.int8(node.children_left) - 1)
            node.best_value = max(value, node.best_value)
            if node.best_value >= node.separator or node.children_left == 0:
                node.terminated = True
                add_board_and_move_to_tt(node, following_move, hash_table)
                # add_one_board_to_tt(node, hash_table)

                # This is being checked at the start of the call also and thus should be removedMoved several dependencies to newer versions
                if not node.parent is None:
                    zero_window_update_node_from_value(node.parent, -node.best_value, node.prev_move, hash_table)





# @njit
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
        # node.next_move = None
        node.next_move_score = None


@njit(GameNode.class_type.instance_type(GameNode.class_type.instance_type))
def spawn_child_from_node(node):
    to_return = zero_window_create_child_from_move(node, node.next_move)

    #If the move used to spawn the child wasn't a move found in the TT move
    if node.children_left != NEXT_MOVE_IS_FROM_TT_VAL:
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




def for_all_white_jitclass_to_array_for_ann(board):
    if board.turn == TURN_WHITE:
        return [board.kings, board.queens, board.rooks, board.bishops, board.knights, board.pawns,
                        board.castling_rights, 0 if board.ep_square is None else BB_SQUARES[np.int32(board.ep_square)],
                        board.occupied_w, board.occupied_b, board.occupied]
    else:
        return [board.kings, board.queens, board.rooks, board.bishops, board.knights, board.pawns,
                        board.castling_rights, 0 if board.ep_square is None else BB_SQUARES[np.int32(board.ep_square)],
                        board.occupied_b, board.occupied_w, board.occupied]



def for_all_white_jitclass_array_to_basic_input_for_anns(board_array):
    to_return = np.empty(shape=[len(board_array),11], dtype=np.uint64)
    black_turn = np.zeros(shape=len(board_array), dtype=np.bool_)
    for j, board in enumerate(board_array):

        to_return[j] = for_all_white_jitclass_to_array_for_ann(board)

        if board.turn == TURN_BLACK:
            black_turn[j] = True


    to_return[black_turn] = vectorized_flip_vertically(to_return[black_turn])

    return to_return.transpose()


def all_white_start_node_evaluations(node_array):
    ann_inputs = for_all_white_jitclass_array_to_basic_input_for_anns(np.array([node.board_state for node in node_array], dtype=np.object))

    def evaluate_and_set():
        results = PREDICTOR(ann_inputs)

        for j, result in enumerate(results):
            node_array[j].best_value = - result


    t = threading.Thread(target=evaluate_and_set)

    t.start()
    return t


@njit
def jitted_set_move_scores_and_next_move(node, scores):
    node.unexplored_move_scores =  scores
    set_up_next_best_move(node)


def start_move_scoring(node_array):
    ann_inputs = for_all_white_jitclass_array_to_basic_input_for_anns(np.array([node.board_state for node in node_array],dtype=np.object))

    #This is taking up a lot of time, a change of the GameNode class to a numpy scalar is something I've been thinking
    #about and which (I believe) would allow for a much faster implementation of this
    move_index_arrays = [MOVE_TO_INDEX_ARRAY[node.unexplored_moves["from_square"], node.unexplored_moves[
        "to_square"]] if node.board_state.turn == TURN_WHITE else 1792 - MOVE_TO_INDEX_ARRAY[
        node.unexplored_moves["from_square"], node.unexplored_moves["to_square"]] for node in
                         node_array]

    size_array = np.array([ary.shape[0] for ary in move_index_arrays])

    for_result_getter = np.stack(
        [np.repeat(np.arange(len(size_array)), size_array), np.concatenate(move_index_arrays, axis=0)], axis=1)

    def set_move_scores():
        results = MOVE_PREDICTOR(ann_inputs,
                                 for_result_getter)

        for node, scores in  zip(node_array, np.split(results,np.cumsum(size_array))):
            jitted_set_move_scores_and_next_move(node, scores)


    t  = threading.Thread(target=set_move_scores)
    t.start()
    return t


@njit(boolean(BoardState.class_type.instance_type))
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


# @njit(nogil=True)#(boolean(GameNode.class_type.instance_type, optional(hash_table_numba_dtype[:,:])),nogil=True)
# def check_and_update_valid_moves(game_node, hash_table):
#     """
#     CURRENTLY CHECKING:
#     1) Draw by the 50-move rule
#     2) Draw by insufficient material
#     3) Draw by stalemate
#     4) Win/loss by checkmate
#     5) Termination by information contained in the TT
#     """
#
#     if game_node.board_state.halfmove_clock >= 50:
#         game_node.best_value = TIE_RESULT_SCORE
#         return True
#
#     if has_insufficient_material(game_node.board_state):
#         game_node.best_value = TIE_RESULT_SCORE
#         return True
#
#     if not hash_table is None:
#         if terminated_from_tt(game_node, hash_table):
#             return True
#
#     legal_move_struct_array = create_legal_move_struct(game_node.board_state, BB_ALL, BB_ALL)
#     if not legal_move_struct_array is None:
#         game_node.unexplored_moves = legal_move_struct_array
#         game_node.children_left = len(legal_move_struct_array)
#     else:
#         if is_in_check(game_node.board_state):
#             game_node.best_value = LOSS_RESULT_SCORE
#         else:
#             game_node.best_value = TIE_RESULT_SCORE
#         return True
#     return False
#
#
# def get_indices_of_terminating_children(node_array, hash_table):
#     """
#     Currently using ThreadPoolExecutor to do this on multiple cores, after using Numba to release the GIL.
#     This will soon be completely vectorized.
#     """
#     MAX_WORKERS = 6 #This number is picked arbitrarily, setting this equal to 12 (how many cores I have available) does not improve speed
#
#     def loop_wrapper(inner_node_array):
#         terminating = np.zeros(shape=len(inner_node_array), dtype=np.bool_)
#         for j in range(len(inner_node_array)):
#             terminating[j] = check_and_update_valid_moves(inner_node_array[j], hash_table)
#         return terminating
#
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         return np.concatenate(list(executor.map(loop_wrapper,np.array_split(node_array, MAX_WORKERS))))






@njit(boolean(GameNode.class_type.instance_type, hash_table_numba_dtype[:]))
def has_no_legal_tt_move(game_node, hash_table):
    """
    Checks if a move is being stored in the transposition table for the given node, and if there is, that the
    move is legal.  If it does find a legal move, it stores the move as the node's next move, sets the
    node's next move score to a constant value, and sets it's children_left to a specified constant to indicate there
    was a legal move found in the tt.  It then returns False if a move is found, or True if not.
    """
    node_entry = hash_table[game_node.board_state.cur_hash & ONES_IN_RELEVANT_BITS]
    if node_entry["depth"] != NO_TT_ENTRY_VALUE:
        if node_entry["entry_hash"] == game_node.board_state.cur_hash:
            if node_entry["stored_from_square"] != NO_TT_MOVE_VALUE:
                tt_move = Move(node_entry["stored_from_square"],
                               node_entry["stored_to_square"],
                               node_entry["stored_promotion"])

                if is_legal_move(game_node.board_state, tt_move):
                    game_node.next_move = tt_move

                    game_node.next_move_score = TT_MOVE_SCORE_VALUE

                    # Indicate that the node has chosen a TT move, and hasn't done a full move generation/scoring
                    game_node.children_left = NEXT_MOVE_IS_FROM_TT_VAL
                    return False
    return True


def single_thread_get_indices_of_terminating_children(node_array, hash_table):
    """
    Gets a mask for the given array to determine which of the given nodes should be terminated (for any of the
    reasons listed below).  If any nodes should terminate, their best_value is set to the value they would terminate
    with, and if not, their legal moves are generated and set.  In the case that a node should not terminate,
    and there is a legal move for the node in the transposition table, it will not generate the legal moves
    for the node, instead it will do what is described in the comments of the function has_no_legal_tt_move
    (store the one move and indicate it hasn't generated all the moves yet).


    CURRENTLY CHECKING:
    1) Draw by the 50-move rule
    2) Draw by insufficient material
    3) Draw by stalemate
    4) Win/loss by checkmate
    5) Termination by information contained in the TT
    """

    terminating = np.zeros(len(node_array),dtype=np.bool_)
    for j, game_node in enumerate(node_array):


        if game_node.board_state.halfmove_clock >= 50:
            game_node.best_value = TIE_RESULT_SCORE
            terminating[j] = True
        elif has_insufficient_material(game_node.board_state):
            game_node.best_value = TIE_RESULT_SCORE
            terminating[j] = True

        elif not hash_table is None and terminated_from_tt(game_node, hash_table):
            terminating[j] = True
        elif has_no_legal_tt_move(game_node, hash_table):
            legal_move_struct_array = create_legal_move_struct(game_node.board_state, BB_ALL, BB_ALL)
            if not legal_move_struct_array is None:
                game_node.unexplored_moves = legal_move_struct_array
                game_node.children_left = len(legal_move_struct_array)
            else:
                if is_in_check(game_node.board_state):
                    game_node.best_value = LOSS_RESULT_SCORE
                else:
                    game_node.best_value = TIE_RESULT_SCORE
                terminating[j] = True

    return terminating

@njit
def depth_zero_should_terminate(game_node, hash_table):
    """
    Tests a given depth zero node for termination, based on the same factors as listed in the comment of the
    single_thread_get_indices_of_terminating_children function.  If the node is terminating, then this function
    will set the value it's terminating with to the node's best_value.
    """
    if game_node.board_state.halfmove_clock >= 50:
        game_node.best_value = TIE_RESULT_SCORE
        return True
    elif has_insufficient_material(game_node.board_state):
        game_node.best_value = TIE_RESULT_SCORE
        return True
    elif terminated_from_tt(game_node, hash_table):
        return True
    elif not has_legal_move(game_node.board_state):
        if is_in_check(game_node.board_state):
            game_node.best_value = LOSS_RESULT_SCORE
        else:
            game_node.best_value = TIE_RESULT_SCORE
        return True

    return False


def depth_zero_should_terminate_array(game_nodes, hash_table):
    return np.array(list(map(lambda x: depth_zero_should_terminate(x, hash_table), game_nodes)), dtype=np.bool_)


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


def update_tree_from_terminating_nodes(node_array, hash_table):
    for j in range(len(node_array)):

        #Since no node being given to this function will have terminated from a child, none of them will have moves to store
        add_one_board_to_tt(node_array[j], hash_table)

        zero_window_update_node_from_value(node_array[j].parent, - node_array[j].best_value, node_array[j].prev_move, hash_table)


@njit(boolean(GameNode.class_type.instance_type))
def generate_moves_for_tt_move_node(game_node):
    """
    Generates the legal moves for a GameNode object when a legal move for the node was found in
    the transposition table (TT), and then was expanded in the same iteration as this function is being run.
    So it generates all of the legal moves except for the move which was already found in the TT.
    It then sets the GameNode's children_left to the actual number of children left,
    as opposed to the indicator value NEXT_MOVE_IS_FROM_TT_VAL which it previously was.
    This function returns True if it discovers more children to explore, and False if not.
    """

    game_node.unexplored_moves = create_legal_move_struct_without_move(game_node.board_state, game_node.next_move, BB_ALL, BB_ALL)

    if game_node.unexplored_moves is None:
        game_node.children_left = 1
        return False
    else:
        game_node.children_left = 1 + len(game_node.unexplored_moves)
        return True


def generate_moves_for_array_of_tt_move_nodes(node_array):
    """
    Returns a mask of the indices of node_array where more moves were found
    """
    return np.array([generate_moves_for_tt_move_node(node) for node in node_array], dtype=np.bool_)






def do_iteration(batch, hash_table, testing=False):
    children = create_child_array(batch)

    depth_zero_children_mask = np.array([True if child.depth == 0 else False for child in children], dtype=np.bool_)

    depth_zero_children = children[depth_zero_children_mask]
    depth_not_zero_children = children[np.logical_not(depth_zero_children_mask)]

    depth_zero_not_scored_mask = np.logical_not(depth_zero_should_terminate_array(depth_zero_children, hash_table))

    if np.any(depth_zero_not_scored_mask):
        evaluation_thread = all_white_start_node_evaluations(depth_zero_children[depth_zero_not_scored_mask])
    else:
        evaluation_thread = None


    child_was_from_tt_move_mask = np.array(
        [True if node.children_left==NEXT_MOVE_IS_FROM_TT_VAL else False for node in batch],
        dtype=np.bool_)

    #(This should likely be put directly into it's only use below to prevent storing it in memory)
    tt_move_nodes_with_more_kids_indices = generate_moves_for_array_of_tt_move_nodes(batch[child_was_from_tt_move_mask])


    non_zero_depth_child_term_indices = single_thread_get_indices_of_terminating_children(depth_not_zero_children, hash_table)
    non_zero_depth_child_not_term_indices = np.logical_not(non_zero_depth_child_term_indices)



    found_no_move_in_tt_indices = np.logical_and(
        non_zero_depth_child_not_term_indices,
        np.array(
            [True if node.children_left!=NEXT_MOVE_IS_FROM_TT_VAL else False for node in depth_not_zero_children],
            dtype=np.bool_))


    TO_SCORE_MOVES_NODES = np.concatenate((
        depth_not_zero_children[found_no_move_in_tt_indices],
        batch[child_was_from_tt_move_mask][tt_move_nodes_with_more_kids_indices]))


    if len(TO_SCORE_MOVES_NODES) != 0:
        move_thread = start_move_scoring(TO_SCORE_MOVES_NODES)
    else:
        move_thread = None


    have_children_left_indices = np.array([True if has_uncreated_child(node) else False for node in batch], dtype=np.bool_)


    if not evaluation_thread is None:
        evaluation_thread.join()


    update_tree_from_terminating_nodes(
        np.concatenate((
            depth_zero_children,
            depth_not_zero_children[non_zero_depth_child_term_indices])),
        hash_table)


    if not move_thread is None:
        move_thread.join()

    return np.concatenate((
        batch[have_children_left_indices],
        depth_not_zero_children[non_zero_depth_child_not_term_indices]))






def do_alpha_beta_search_with_bins(root_game_node, max_batch_size, bins_to_use, hash_table=None, print_info=False, full_testing=False):
    open_node_holder = PriorityBins(bins_to_use, max_batch_size, num_workers_to_use=12, search_extra_ratio=1.2, testing=full_testing) #This will likely be put outside of this function long term

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
                    if to_insert[j].depth == 0:
                        print("Found a node with depth zero being inserted into global nodes!")



        if print_info or full_testing:
            print("Iteration", iteration_counter, "completed producing", len(to_insert), "nodes to insert.")

        try_num_getting_batch = 1
        while True:
        #     print(len(open_node_holder))
            next_batch = open_node_holder.insert_batch_and_get_next_batch(to_insert, custom_max_nodes=max_batch_size*try_num_getting_batch)

            to_insert = np.array([])
            if len(next_batch) != 0 or (len(next_batch) == 0 and len(open_node_holder) == 0):
                break
            else:
                try_num_getting_batch += 1


        # if len(to_insert) != 0 or not open_node_holder.is_empty():
        #     next_batch = open_node_holder.new_insert_batch_and_get_next_batch(to_insert)
        # else:
        #     next_batch = []

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

    if print_info or full_testing:
        time_taken_without_tt = time.time() - start_time
        print("Total time taken not including table_creation:", time_taken_without_tt)
        print("Number of iterations taken", iteration_counter, "in total time", time_taken_without_tt)
        print("Average time per iteration:", time_taken_without_tt / iteration_counter)
        print("Total nodes computed:", total_nodes_computed)
        print("Nodes evaluated per second:", total_nodes_computed/time_taken_without_tt)
        print("Total nodes given to linked list for insert", given_for_insert)

    return root_game_node.best_value


# @njit(float32(GameNode.class_type.instance_type, float32, float32, nb.int8))
# def basic_do_alpha_beta(game_node, alpha, beta, color):
#     """
#     A simple alpha-beta algorithm to test my code.
#
#     NOTES:
#     1) This is PAINFULLY slow
#     """
#     has_move = False
#     for _ in generate_legal_moves(game_node.board_state, BB_ALL, BB_ALL):
#         has_move = True
#         break
#
#     if not has_move:
#         if is_in_check(game_node.board_state):
#             if color == 1:
#                 return MAX_FLOAT32_VAL
#             return MIN_FLOAT32_VAL
#         return 0
#
#     if game_node.depth == 0:
#         return color *  PREDICTOR([np.int64(np.uint64(game_node.board_state.kings))],
#                                   [np.int64(np.uint64(game_node.board_state.queens))],
#                                   [np.int64(np.uint64(game_node.board_state.rooks))],
#                                   [np.int64(np.uint64(game_node.board_state.bishops))],
#                                   [np.int64(np.uint64(game_node.board_state.knights))],
#                                   [np.int64(np.uint64(game_node.board_state.pawns))],
#                                   [np.int64(np.uint64(game_node.board_state.castling_rights))],
#                                   [np.int64(np.uint64(game_node.board_state.ep_square)) if not game_node.board_state.ep_square is None else 255],
#                                   [np.int64(np.uint64(game_node.board_state.occupied_w))],
#                                   [np.int64(np.uint64(game_node.board_state.occupied_b))],
#                                   [np.int64(np.uint64(game_node.board_state.occupied))],
#                                   [game_node.board_state.turn])
#
#
#
#         # start_node_evaluations(np.array([game_node])).join()
#         # return color *  game_node.best_value
#
#     best_value = MIN_FLOAT32_VAL
#     for move in generate_legal_moves(game_node.board_state, BB_ALL, BB_ALL):
#         v = - basic_do_alpha_beta(zero_window_create_child_from_move(game_node, move), -beta, -alpha, -color)
#         best_value = max(best_value, v)
#         alpha = max(alpha, v)
#         if alpha >= beta:
#             break
#
#     return best_value





def set_up_root_search_tree_node(fen, depth, seperator):
    root_node = create_game_node_from_fen(fen, depth, seperator)
    root_node.unexplored_moves = create_legal_move_struct(root_node.board_state, BB_ALL, BB_ALL)
    root_node.children_left = len(root_node.unexplored_moves)
    root_node_as_array = np.array([root_node])
    thread = start_move_scoring(root_node_as_array)
    thread.join()
    return root_node


def mtd_f(fen, depth, first_guess, min_window_to_confirm, guess_increment=.5, search_batch_size=1000, bins_to_use=None, hash_table=None, print_info=False):
    if hash_table is None:
        hash_table = get_empty_hash_table()

    if bins_to_use is None:
        bins_to_use = np.arange(15, -15, -.025)


    counter = 0
    cur_guess = first_guess
    upper_bound = MAX_FLOAT32_VAL
    lower_bound = MIN_FLOAT32_VAL
    while upper_bound -  min_window_to_confirm > lower_bound:

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
        cur_root_node = set_up_root_search_tree_node(fen, depth, beta - FLOAT32_EPS)


        cur_guess = do_alpha_beta_search_with_bins(cur_root_node, search_batch_size, bins_to_use, hash_table=hash_table, print_info=depth==5)
        if cur_guess < beta:
            upper_bound = cur_guess
        else:
            lower_bound = cur_guess

        if print_info:
            print("Finished iteration %d with lower and upper bounds (%f,%f)" % (counter+1, lower_bound, upper_bound))

        counter += 1

    root_tt_entry = hash_table[np.uint64(cur_root_node.board_state.cur_hash) & ONES_IN_RELEVANT_BITS]
    tt_move = chess.Move(root_tt_entry["stored_from_square"],
                   root_tt_entry["stored_to_square"],
                   root_tt_entry["stored_promotion"])

    return cur_guess, tt_move, hash_table



def iterative_deepening_mtd_f(fen, depths_to_search, batch_sizes, min_windows_to_confirm, first_guess=0, guess_increments=None, bin_sets=None, hash_table=None):
    if hash_table is None:
        starting_time = time.time()
        hash_table = get_empty_hash_table()
        # print("Hash table created in time:", time.time() - starting_time)

    if bin_sets is None:
        bin_sets = (np.arange(15, -15, -.025) for _ in range(len(depths_to_search)))

    if guess_increments is None:
        guess_increments = [1]*len(depths_to_search)

    for depth, batch_size, window, increment, bins in zip(depths_to_search, batch_sizes, min_windows_to_confirm, guess_increments, bin_sets):
        first_guess, tt_move, hash_table = mtd_f(fen,
                                                 depth,
                                                 first_guess,
                                                 window,
                                                 guess_increment=increment,
                                                 search_batch_size=batch_size,
                                                 bins_to_use=bins,
                                                 hash_table=hash_table)

    return first_guess






if __name__ == "__main__":
    print("Done compiling and starting run.\n")

    # starting_time = time.time()
    # perft_results = jitted_perft_test(create_board_state_from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"), 4)
    # print("Depth 4 perft test results:", perft_results, "in time", time.time()-starting_time, "\n")

    FEN_TO_TEST = "r2q1rk1/pbpnbppp/1p2pn2/3pN3/2PP4/1PN3P1/P2BPPBP/R2Q1RK1 w - - 10 11"#"rn1q1rk1/pbp1bppp/1p2pn2/3pN3/2PP4/1P4P1/P2BPPBP/RN1Q1RK1 w - - 8 10"#"  # "rn1qkb1r/p1pp1ppp/bp2pn2/8/2PP4/5NP1/PP2PP1P/RNBQKB1R w KQkq - 1 5"
    DEPTH_OF_SEARCH = 4
    MAX_BATCH_LIST =  10*[250,500, 1000, 2000]# [j**2 for j in range(40,0,-2)] + [5000]
    SEPERATING_VALUE = 0
    print("SPERATING VALUE:", SEPERATING_VALUE)
    BINS_TO_USE = np.arange(20, -20, -.025)#np.arange(15, -15, -.01)

    start_time = time.time()
    print(mtd_f(FEN_TO_TEST,DEPTH_OF_SEARCH,0,.001, search_batch_size=10000)[0],"found by mtd(f)", time.time() - start_time)

    starting_time = time.time()
    print(mtd_f(FEN_TO_TEST, DEPTH_OF_SEARCH, SEPERATING_VALUE, .01), "found by mtd(f) in time", time.time()-starting_time)


    starting_time = time.time()
    for j in MAX_BATCH_LIST:
        root_node = set_up_root_search_tree_node(FEN_TO_TEST, DEPTH_OF_SEARCH, SEPERATING_VALUE)
        print("Root node starting possible moves:", root_node.children_left)
        starting_time = time.time()
        cur_value = do_alpha_beta_search_with_bins(root_node, j, BINS_TO_USE, print_info=True, full_testing=False)
        print("Root node children_left after search:", root_node.children_left)
        if cur_value == MIN_FLOAT32_VAL:
            print("MIN FLOAT VALUE")
        elif cur_value == LOSS_RESULT_SCORE:
            print("LOSS RESULT SCORE")
        print("Value:", cur_value, "found with max batch", j, "in time:", time.time() - starting_time)
        print()


    # print("Total white move sets evaluated:", ANN_MOVE_EVALUATOR_WHITE.total_evaluated_nodes, "with an average batch size of:", ANN_MOVE_EVALUATOR_WHITE.total_evaluated_nodes / ANN_MOVE_EVALUATOR_WHITE.total_evaluations)
    # print("Total black move sets evaluated:", ANN_MOVE_EVALUATOR_BLACK.total_evaluated_nodes, "with an average batch size of:", ANN_MOVE_EVALUATOR_BLACK.total_evaluated_nodes / ANN_MOVE_EVALUATOR_BLACK.total_evaluations)
    # print("Total boards scored:", ANN_BOARD_EVALUATOR.total_evaluated_nodes, "with an average batch size of:", ANN_BOARD_EVALUATOR.total_evaluated_nodes / ANN_BOARD_EVALUATOR.total_evaluations)


    # root_node = create_game_node_from_fen(FEN_TO_TEST, DEPTH_OF_SEARCH ,-1)
    # for j in range(1):
    #     starting_time = time.time()
    #     print("Basic negamax with alpha-beta pruning resulted in:",
    #           basic_do_alpha_beta(root_node, MIN_FLOAT32_VAL, MAX_FLOAT32_VAL, 1), "in time:", time.time() - starting_time)


    CLOSER()
    MOVE_CLOSER()