import numpy as np
import numba as nb

from numba import njit

import chess
import itertools
import functools

from chess.polyglot import POLYGLOT_RANDOM_ARRAY, zobrist_hash

from numba import cffi_support
from cffi import FFI

ffi = FFI()

import khash_numba._khash_ffi as khash_ffi

cffi_support.register_module(khash_ffi)

khash_init = khash_ffi.lib.khash_int2int_init
khash_get = khash_ffi.lib.khash_int2int_get
khash_set = khash_ffi.lib.khash_int2int_set
khash_destroy = khash_ffi.lib.khash_int2int_destroy


@njit
def create_index_table(ids):
    table = khash_init()
    for j in range(len(ids)):
        khash_set(table, ids[j], j)
    return table


def get_table_and_array_for_set_of_dicts(dicts):
    unique_keys = sorted(set(itertools.chain.from_iterable(dicts)))

    # The sorted is so that the index of 0 will always be 0
    index_lookup_table = create_index_table(np.array(sorted([np.uint64(key) for key in unique_keys]), dtype=np.uint64))

    array = np.zeros(shape=[len(dicts), len(unique_keys)], dtype=np.uint64)

    for square_num, dict in enumerate(dicts):
        for key, value in dict.items():
            index_to_set = khash_get(ffi.cast("void *", index_lookup_table), np.uint64(key), np.uint64(0))
            array[square_num][index_to_set] = np.uint64(value)

    return index_lookup_table, array


numpy_move_dtype = np.dtype([("from_square", np.uint8), ("to_square", np.uint8), ("promotion", np.uint8)])
move_type = nb.from_dtype(numpy_move_dtype)

RANDOM_ARRAY = np.array(POLYGLOT_RANDOM_ARRAY, dtype=np.uint64)

BB_DIAG_MASKS = np.array(chess.BB_DIAG_MASKS, dtype=np.uint64)
BB_FILE_MASKS = np.array(chess.BB_FILE_MASKS, dtype=np.uint64)
BB_RANK_MASKS = np.array(chess.BB_RANK_MASKS, dtype=np.uint64)

DIAG_ATTACK_INDEX_LOOKUP_TABLE, DIAG_ATTACK_ARRAY = get_table_and_array_for_set_of_dicts(chess.BB_DIAG_ATTACKS)
FILE_ATTACK_INDEX_LOOKUP_TABLE, FILE_ATTACK_ARRAY = get_table_and_array_for_set_of_dicts(chess.BB_FILE_ATTACKS)
RANK_ATTACK_INDEX_LOOKUP_TABLE, RANK_ATTACK_ARRAY = get_table_and_array_for_set_of_dicts(chess.BB_RANK_ATTACKS)

BB_KNIGHT_ATTACKS = np.array(chess.BB_KNIGHT_ATTACKS, dtype=np.uint64)
BB_KING_ATTACKS = np.array(chess.BB_KING_ATTACKS, dtype=np.uint64)

BB_PAWN_ATTACKS = np.array(chess.BB_PAWN_ATTACKS, dtype=np.uint64)

BB_RAYS = np.array(chess.BB_RAYS, dtype=np.uint64)
BB_BETWEEN = np.array(chess.BB_BETWEEN, dtype=np.uint64)

MIN_FLOAT32_VAL = np.finfo(np.float32).min
MAX_FLOAT32_VAL = np.finfo(np.float32).max
ALMOST_MIN_FLOAT_32_VAL = np.nextafter(MIN_FLOAT32_VAL, MAX_FLOAT32_VAL)
ALMOST_MAX_FLOAT_32_VAL = np.nextafter(MAX_FLOAT32_VAL, MIN_FLOAT32_VAL)


# The maximum number of moves which can be stored/assessed by the engine
MAX_MOVES_LOOKED_AT = 100

# The maximum depth the engine is allowed to go
MAX_SEARCH_DEPTH = 100

# This value is used for indicating that a given node has/had a legal move in the transposition table that it will/would
# expand prior to it's full move generation and scoring.
NEXT_MOVE_IS_FROM_TT_VAL = np.uint8(254)

# This value is used for indicating that the next move index of a board is actually a dummy variable, and there are no more moves left
NO_MORE_MOVES_VALUE = np.uint8(255)

# This value is used for indicating that a move in a transposition table entry is not being stored.
NO_TT_MOVE_VALUE = np.uint8(255)

# This value is used for indicating that no entry in the transposition table exists for that hash.  It is stored as the
# entries depth.
NO_TT_ENTRY_VALUE = np.uint8(255)

# This value is the value used to assigned a node who's next move was found in the TT to the desired bin.
TT_MOVE_SCORE_VALUE = ALMOST_MAX_FLOAT_32_VAL

# This value is used when there is no current ep square.  It (obviously) does not indicate that square 0 is an ep square
NO_EP_SQUARE = np.uint8(0)

TIE_RESULT_SCORE = np.float32(0)

# The win/loss arrays are such that the magnitudes of the win/loss decrease as the index in the array increases.
# This is so depth can be used to index the arrays
WIN_RESULT_SCORES = np.full(MAX_SEARCH_DEPTH, np.nextafter(ALMOST_MAX_FLOAT_32_VAL, MIN_FLOAT32_VAL))
LOSS_RESULT_SCORES = np.full(MAX_SEARCH_DEPTH, np.nextafter(ALMOST_MIN_FLOAT_32_VAL, MAX_FLOAT32_VAL))

for j in range(1, MAX_SEARCH_DEPTH):
    WIN_RESULT_SCORES[j] = np.nextafter(WIN_RESULT_SCORES[j - 1], MIN_FLOAT32_VAL)
    LOSS_RESULT_SCORES[j] = np.nextafter(LOSS_RESULT_SCORES[j - 1], MAX_FLOAT32_VAL)



SIZE_EXPONENT_OF_TWO_FOR_TT_INDICES = np.uint8(30)  # This needs to be picked precisely
TT_HASH_MASK = np.uint64(2 ** (SIZE_EXPONENT_OF_TWO_FOR_TT_INDICES) - 1)


COLORS = [WHITE, BLACK] = np.array([1, 0], dtype=np.uint8)
TURN_COLORS = [TURN_WHITE, TURN_BLACK] = [True, False]
PIECE_TYPES = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] = np.arange(1, 7, dtype=np.uint8)


SQUARES = [
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8] = np.arange(64, dtype=np.uint8)

SQUARES_180 = SQUARES ^ 0x38

BB_VOID = np.uint64(0)
BB_ALL = np.uint64(0xffffffffffffffff)

BB_SQUARES = [
    BB_A1, BB_B1, BB_C1, BB_D1, BB_E1, BB_F1, BB_G1, BB_H1,
    BB_A2, BB_B2, BB_C2, BB_D2, BB_E2, BB_F2, BB_G2, BB_H2,
    BB_A3, BB_B3, BB_C3, BB_D3, BB_E3, BB_F3, BB_G3, BB_H3,
    BB_A4, BB_B4, BB_C4, BB_D4, BB_E4, BB_F4, BB_G4, BB_H4,
    BB_A5, BB_B5, BB_C5, BB_D5, BB_E5, BB_F5, BB_G5, BB_H5,
    BB_A6, BB_B6, BB_C6, BB_D6, BB_E6, BB_F6, BB_G6, BB_H6,
    BB_A7, BB_B7, BB_C7, BB_D7, BB_E7, BB_F7, BB_G7, BB_H7,
    BB_A8, BB_B8, BB_C8, BB_D8, BB_E8, BB_F8, BB_G8, BB_H8
] = np.array([1 << sq for sq in SQUARES], dtype=np.uint64)

BB_CORNERS = BB_A1 | BB_H1 | BB_A8 | BB_H8

BB_LIGHT_SQUARES = np.uint64(0x55aa55aa55aa55aa)
BB_DARK_SQUARES = np.uint64(0xaa55aa55aa55aa55)


BB_FILES = [
    BB_FILE_A,
    BB_FILE_B,
    BB_FILE_C,
    BB_FILE_D,
    BB_FILE_E,
    BB_FILE_F,
    BB_FILE_G,
    BB_FILE_H
] = np.array([0x0101010101010101 << i for i in range(8)], dtype=np.uint64)

BB_RANKS = [
    BB_RANK_1,
    BB_RANK_2,
    BB_RANK_3,
    BB_RANK_4,
    BB_RANK_5,
    BB_RANK_6,
    BB_RANK_7,
    BB_RANK_8
] = np.array([0xff << (8 * i) for i in range(8)], dtype=np.uint64)


BB_BACKRANKS = BB_RANK_1 | BB_RANK_8

INITIAL_BOARD_FEN = chess.STARTING_FEN

def generate_move_to_enumeration_dict():
    """
    Generates a dictionary where the keys are (from_square, to_square) and their values are the move number
    that move has been assigned.  It is done in a way such that for move number N from board B, if you were to flip B
    vertically, the same move would have number 1792-N. (there are 1792 moves recognized)


    IMPORTANT NOTES:
    1) This ignores the fact that not all pawn promotions are the same, this effects the number of logits
    in the move scoring ANN
    """
    possible_moves = {}

    board = chess.Board('8/8/8/8/8/8/8/8 w - - 0 1')
    for square in chess.SQUARES[:len(SQUARES) // 2]:
        for piece in [chess.Piece(chess.KNIGHT, True), chess.Piece(chess.QUEEN, True)]:
            board.set_piece_at(square, piece)
            for move in board.generate_legal_moves():
                if possible_moves.get((move.from_square, move.to_square)) is None:
                    possible_moves[move.from_square, move.to_square] = len(possible_moves)

            board.remove_piece_at(square)

    switch_square_fn = lambda x: x ^ 0x38

    total_possible_moves = len(possible_moves) * 2 - 1

    for (from_square, to_square), move_num in list(possible_moves.items()):
        possible_moves[switch_square_fn(from_square), switch_square_fn(to_square)] = total_possible_moves - move_num

    return possible_moves


MOVE_TO_INDEX_ARRAY = np.zeros(shape=[64, 64], dtype=np.int32)
OLD_REVERSED_MOVE_TO_INDEX_ARRAY = np.zeros(shape=[64, 64], dtype=np.int32)
REVERSED_MOVE_TO_INDEX_ARRAY = np.zeros(shape=[64, 64], dtype=np.int32)

dict_keys, dict_values = list(zip(*generate_move_to_enumeration_dict().items()))
dict_keys = np.array(dict_keys)

MOVE_TO_INDEX_ARRAY[dict_keys[:,0], dict_keys[:,1]] = dict_values

reversed_keys = dict_keys ^ 0x38
REVERSED_MOVE_TO_INDEX_ARRAY[reversed_keys[:,0], reversed_keys[:,1]] = dict_values


REVERSED_EP_LOOKUP_ARRAY = SQUARES_180.copy()
REVERSED_EP_LOOKUP_ARRAY[0] = NO_EP_SQUARE


def power_set(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))

flip_vert_const_1 = np.uint64(0x00FF00FF00FF00FF)
flip_vert_const_2 = np.uint64(0x0000FFFF0000FFFF)

@nb.vectorize([nb.uint64(nb.uint64)])
def vectorized_flip_vertically(bb):
    bb = ((bb >> 8) & flip_vert_const_1) | ((bb & flip_vert_const_1) << 8)
    bb = ((bb >> 16) & flip_vert_const_2) | ((bb & flip_vert_const_2) << 16)
    bb = (bb >> 32) | (bb << 32)
    return bb

def get_castling_lookup_tables():
    possible_castling_rights = np.zeros(2 ** 4, dtype=np.uint64)
    for j, set in enumerate(power_set([BB_A1, BB_H1, BB_A8, BB_H8])):
        possible_castling_rights[j] = np.uint64(functools.reduce(lambda x, y: x | y, set, np.uint64(0)))

    white_turn_castling_tables = create_index_table(possible_castling_rights)
    black_turn_castling_tables = create_index_table(vectorized_flip_vertically(possible_castling_rights))

    return white_turn_castling_tables, black_turn_castling_tables, possible_castling_rights


WHITE_CASTLING_RIGHTS_LOOKUP_TABLE, BLACK_CASTLING_RIGHTS_LOOKUP_TABLE, POSSIBLE_CASTLING_RIGHTS = get_castling_lookup_tables()
