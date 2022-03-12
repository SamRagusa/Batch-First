import numpy as np

import chess
from chess.polyglot import POLYGLOT_RANDOM_ARRAY, zobrist_hash
from chess import BB_KNIGHT_ATTACKS as OLD_BB_KNIGHT_ATTACKS
from chess import BB_KING_ATTACKS as OLD_BB_KING_ATTACKS
from chess import BB_PAWN_ATTACKS as OLD_BB_PAWN_ATTACKS
from chess import BB_DIAG_MASKS as OLD_BB_DIAG_MASKS
from chess import BB_FILE_MASKS as OLD_BB_FILE_MASKS
from chess import BB_RANK_MASKS as OLD_BB_RANK_MASKS
from chess import BB_DIAG_ATTACKS as OLD_BB_DIAG_ATTACKS
from chess import BB_FILE_ATTACKS as OLD_BB_FILE_ATTACKS
from chess import BB_RANK_ATTACKS as OLD_BB_RANK_ATTACKS
from chess import BB_RAYS as OLD_BB_RAYS
from chess import BB_BETWEEN as OLD_BB_BETWEEN

import numba as nb
from numba import int32, int64, uint8, uint16, uint64, boolean, void, deferred_type, optional
from numba import jit, njit, vectorize
from numba.typed import Dict, List
from numba.experimental import jitclass
# from numba import cffi_support
from cffi import FFI
import time
# import types
from numba import types

ffi = FFI()

from collections import OrderedDict


from chess import BB_DIAG_ATTACKS as OLD_BB_DIAG_ATTACKS
from chess import BB_FILE_ATTACKS as OLD_BB_FILE_ATTACKS
from chess import BB_RANK_ATTACKS as OLD_BB_RANK_ATTACKS

# @jit
def test():
    example_dict = Dict.empty(
        key_type=types.uint64,
        value_type=types.uint64,
    )
    diag_lst = List.empty_list(nb.typeof(example_dict))#[]
    file_lst = List.empty_list(nb.typeof(example_dict))#[]
    rank_lst = List.empty_list(nb.typeof(example_dict))#[]
    # print(len(OLD_BB_DIAG_ATTACKS), len(OLD_BB_FILE_ATTACKS), len(OLD_BB_RANK_ATTACKS))
    for j in range(64):
        diag_lst.append(Dict.empty(types.uint64, types.uint64))
        file_lst.append(Dict.empty(types.uint64, types.uint64))
        rank_lst.append(Dict.empty(types.uint64, types.uint64))

    for j, cur_dict in enumerate(OLD_BB_DIAG_ATTACKS):
        # OLD_BB_DIAG_ATTACKS[j] = Dict.empty(
        #     key_type=types.uint64,
        #     value_type=types.uint64,
        # )
        for k, v in cur_dict.items():
            diag_lst[j][np.uint64(k)] = np.uint64(v)

        # diag_lst.append(OLD_BB_DIAG_ATTACKS[j])

    for j, cur_dict in enumerate(OLD_BB_FILE_ATTACKS):
        # OLD_BB_FILE_ATTACKS[j] = Dict.empty(
        #     key_type=types.uint64,
        #     value_type=types.uint64,
        # )
        for k, v in cur_dict.items():
            file_lst[j][np.uint64(k)] = np.uint64(v)
        # file_lst.append(OLD_BB_FILE_ATTACKS[j])

    for j, cur_dict in enumerate(OLD_BB_RANK_ATTACKS):
        # OLD_BB_RANK_ATTACKS[j] = Dict.empty(
        #     key_type=types.uint64,
        #     value_type=types.uint64,
        # )
        for k, v in cur_dict.items():
            rank_lst[j][np.uint64(k)] = np.uint64(v)
        # rank_lst.append(OLD_BB_RANK_ATTACKS[j])

    return diag_lst, file_lst, rank_lst
    # return OLD_BB_DIAG_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_RANK_ATTACKS

OLD_BB_DIAG_ATTACKS, OLD_BB_FILE_ATTACKS,OLD_BB_RANK_ATTACKS =  test()
print(list(OLD_BB_RANK_ATTACKS[4].keys()))

# import khash_numba._khash_ffi as khash_ffi
#
# cffi_support.register_module(khash_ffi)
#
# khash_init = khash_ffi.lib.khash_int2int_init
# khash_get = khash_ffi.lib.khash_int2int_get
# khash_set = khash_ffi.lib.khash_int2int_set
# khash_destroy = khash_ffi.lib.khash_int2int_destroy


numpy_move_dtype = np.dtype([("from_square", np.uint8), ("to_square", np.uint8), ("promotion", np.uint8)])
move_type = nb.from_dtype(numpy_move_dtype)

MAX_MOVES_LOOKED_AT = 100
EMPTY_MOVE_ARRAY = np.zeros([MAX_MOVES_LOOKED_AT], numpy_move_dtype)


# @njit
# def create_index_table(ids):
#     table = khash_init()
#
#     for j in range(ids.shape[0]):
#         khash_set(table, ids[j], j)
#     return table
#
#
#
# def get_table_and_array_for_set_of_dicts(dicts):
#     print(dicts)
#     for j in range(len(dicts)):
#         if not j:
#             all_keys = np.array([i for i in dicts[j].keys()], dtype=np.uint64)
#         else:
#             all_keys = np.concatenate([all_keys, np.array([i for i in dicts[j].keys()], dtype=np.uint64)])
#
#     unique_keys = sorted(set(all_keys))
#
#     # The sorted is so that the index of 0 will always be 0
#     index_lookup_table = create_index_table(np.array(sorted([np.uint64(key) for key in unique_keys]), dtype=np.uint64))
#
#     array = np.zeros(shape=[len(dicts), len(unique_keys)], dtype=np.uint64)
#
#     for square_num, dict in enumerate(dicts):
#         for key, value in dict.items():
#             array[square_num][
#                 khash_get(ffi.cast("void *", index_lookup_table), np.uint64(key), np.uint64(0))] = np.uint64(value)
#
#     return index_lookup_table, array




RANDOM_ARRAY = np.array(POLYGLOT_RANDOM_ARRAY, dtype=np.uint64)


BB_DIAG_MASKS = np.array(OLD_BB_DIAG_MASKS, dtype=np.uint64)
BB_FILE_MASKS = np.array(OLD_BB_FILE_MASKS, dtype=np.uint64)
BB_RANK_MASKS = np.array(OLD_BB_RANK_MASKS, dtype=np.uint64)


# DIAG_ATTACK_INDEX_LOOKUP_TABLE, DIAG_ATTACK_ARRAY = get_table_and_array_for_set_of_dicts(OLD_BB_DIAG_ATTACKS)
# FILE_ATTACK_INDEX_LOOKUP_TABLE, FILE_ATTACK_ARRAY = get_table_and_array_for_set_of_dicts(OLD_BB_FILE_ATTACKS)
# RANK_ATTACK_INDEX_LOOKUP_TABLE, RANK_ATTACK_ARRAY = get_table_and_array_for_set_of_dicts(OLD_BB_RANK_ATTACKS)



BB_KNIGHT_ATTACKS = np.array(OLD_BB_KNIGHT_ATTACKS, dtype=np.uint64)
BB_KING_ATTACKS = np.array(OLD_BB_KING_ATTACKS, dtype=np.uint64)

BB_PAWN_ATTACKS = np.array(OLD_BB_PAWN_ATTACKS, dtype=np.uint64)


BB_RAYS = np.array(OLD_BB_RAYS, dtype=np.uint64)
BB_BETWEEN = np.array(OLD_BB_BETWEEN, dtype=np.uint64)




board_state_spec = OrderedDict()

board_state_spec["pawns"] = uint64
board_state_spec["knights"] = uint64
board_state_spec["bishops"] = uint64
board_state_spec["rooks"] = uint64
board_state_spec["queens"] = uint64
board_state_spec["kings"] = uint64

board_state_spec["occupied_w"] = uint64
board_state_spec["occupied_b"] = uint64
board_state_spec["occupied"] = uint64

board_state_spec["turn"] = boolean
board_state_spec["castling_rights"] = uint64
board_state_spec["ep_square"] = optional(uint8)

board_state_spec["halfmove_clock"] = uint8

board_state_spec["cur_hash"] = uint64


@jitclass(board_state_spec)
class BoardState:
    def __init__(self, pawns, knights, bishops, rooks, queens, kings, occupied_w, occupied_b, occupied, turn,
                 castling_rights, ep_square, halfmove_clock, cur_hash):
        self.pawns = pawns
        self.knights = knights
        self.bishops = bishops
        self.rooks = rooks
        self.queens = queens
        self.kings = kings

        self.occupied_w = occupied_w
        self.occupied_b = occupied_b
        self.occupied = occupied

        self.turn = turn
        self.castling_rights = castling_rights
        self.ep_square = ep_square
        self.halfmove_clock = halfmove_clock

        self.cur_hash = cur_hash





@njit#(BoardState.class_type.instance_type(BoardState.class_type.instance_type))
def copy_board_state(board_state):
    return BoardState(board_state.pawns,
                      board_state.knights,
                      board_state.bishops,
                      board_state.rooks,
                      board_state.queens,
                      board_state.kings,
                      board_state.occupied_w,
                      board_state.occupied_b,
                      board_state.occupied,
                      board_state.turn,
                      board_state.castling_rights,
                      board_state.ep_square,
                      board_state.halfmove_clock,
                      board_state.cur_hash)

def flipped_board_state(board_state):
    return BoardState(flip_vertically(board_state.pawns),
                      flip_vertically(board_state.knights),
                      flip_vertically(board_state.bishops),
                      flip_vertically(board_state.rooks),
                      flip_vertically(board_state.queens),
                      flip_vertically(board_state.kings),
                      flip_vertically(board_state.occupied_w),
                      flip_vertically(board_state.occupied_b),
                      flip_vertically(board_state.occupied),
                      not board_state.turn,
                      flip_vertically(board_state.castling_rights),
                      None if board_state.ep_square is None else square_mirror(board_state.ep_square),
                      board_state.halfmove_clock,
                      board_state.cur_hash)



move_spec = OrderedDict()
move_spec["from_square"] = uint8
move_spec["to_square"] = uint8
move_spec["promotion"] = uint8


@jitclass(move_spec)
class Move:
    """
    Represents a move from a square to a square and possibly the promotion
    piece type.

    Drops and null moves are supported.

    COMMENT NO LONGER ACCURATE
    """

    def __init__(self, from_square, to_square, promotion):
        self.from_square = from_square
        self.to_square = to_square
        self.promotion = promotion


@njit#(boolean(Move.class_type.instance_type, Move.class_type.instance_type))
def move_objects_equal(move1, move2):
    """
    Checks if the two given move objects are equal.
    """
    if move1.from_square == move2.from_square and move1.to_square == move2.to_square and move1.promotion == move2.promotion:
        return True
    return False


COLORS = [WHITE, BLACK] = [1, 0]
TURN_COLORS = [TURN_WHITE, TURN_BLACK] = [True, False]
COLOR_NAMES = ["black", "white"]

PIECE_TYPES = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] = range(1, 7)
PIECE_SYMBOLS = ["", "p", "n", "b", "r", "q", "k"]
PIECE_NAMES = ["", "pawn", "knight", "bishop", "rook", "queen", "king"]

old_squares = [
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8] = [np.uint8(num) for num in range(64)]

SQUARES = np.array(old_squares, dtype=np.uint8)

SQUARES_180 = [chess.square_mirror(sq) for sq in SQUARES]

BB_VOID = np.uint64(0)
BB_ALL = np.uint64(0xffffffffffffffff)

normal_bb_squares = [
    BB_A1, BB_B1, BB_C1, BB_D1, BB_E1, BB_F1, BB_G1, BB_H1,
    BB_A2, BB_B2, BB_C2, BB_D2, BB_E2, BB_F2, BB_G2, BB_H2,
    BB_A3, BB_B3, BB_C3, BB_D3, BB_E3, BB_F3, BB_G3, BB_H3,
    BB_A4, BB_B4, BB_C4, BB_D4, BB_E4, BB_F4, BB_G4, BB_H4,
    BB_A5, BB_B5, BB_C5, BB_D5, BB_E5, BB_F5, BB_G5, BB_H5,
    BB_A6, BB_B6, BB_C6, BB_D6, BB_E6, BB_F6, BB_G6, BB_H6,
    BB_A7, BB_B7, BB_C7, BB_D7, BB_E7, BB_F7, BB_G7, BB_H7,
    BB_A8, BB_B8, BB_C8, BB_D8, BB_E8, BB_F8, BB_G8, BB_H8
] = [np.uint64(1 << sq) for sq in SQUARES]

BB_SQUARES = np.array(normal_bb_squares, dtype=np.uint64)

BB_CORNERS = BB_A1 | BB_H1 | BB_A8 | BB_H8

BB_LIGHT_SQUARES = np.uint64(0x55aa55aa55aa55aa)
BB_DARK_SQUARES = np.uint64(0xaa55aa55aa55aa55)

old_bb_files = [
    BB_FILE_A,
    BB_FILE_B,
    BB_FILE_C,
    BB_FILE_D,
    BB_FILE_E,
    BB_FILE_F,
    BB_FILE_G,
    BB_FILE_H
] = [np.uint64(0x0101010101010101 << i) for i in range(8)]

BB_FILES = np.array(old_bb_files, dtype=np.uint64)

old_bb_ranks = [
    BB_RANK_1,
    BB_RANK_2,
    BB_RANK_3,
    BB_RANK_4,
    BB_RANK_5,
    BB_RANK_6,
    BB_RANK_7,
    BB_RANK_8
] = [np.uint64(0xff << (8 * i)) for i in range(8)]

BB_RANKS = np.array(old_bb_ranks, dtype=np.uint64)

BB_BACKRANKS = BB_RANK_1 | BB_RANK_8

INITIAL_BOARD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


flip_vert_const_1 = np.uint64(0x00FF00FF00FF00FF)
flip_vert_const_2 = np.uint64(0x0000FFFF0000FFFF)

@vectorize([uint64(uint64)])
def vectorized_flip_vertically(bb):
    bb = ((bb >>  8) & flip_vert_const_1) | ((bb & flip_vert_const_1) <<  8)
    bb = ((bb >> 16) & flip_vert_const_2) | ((bb & flip_vert_const_2) << 16)
    bb = ( bb >> 32) | ( bb << 32)
    return bb

@njit(uint64(uint64))
def flip_vertically(bb):
    bb = ((bb >>  8) & flip_vert_const_1) | ((bb & flip_vert_const_1) <<  8)
    bb = ((bb >> 16) & flip_vert_const_2) | ((bb & flip_vert_const_2) << 16)
    bb = ( bb >> 32) | ( bb << 32)
    return bb


# THIS IS VERY SLOW AND A TEMPORARY IMPLEMENTATION (feel free to replace it!)
@njit(uint8(uint64))
def msb(n):
    r = 0
    n = n >> 1
    while n:
        r += 1
        n = n >> 1
    return r


# THIS SEEMS VERY SLOW AND A TEMPORARY IMPLEMENTATION (feel free to replace it!)
@njit
def scan_reversed(bb):
    for index in range(BB_SQUARES.shape[0]):
        if bb & BB_SQUARES[index]:
            yield np.uint8(index)
            bb ^= BB_SQUARES[index]
    return



k1 = np.uint64(0x5555555555555555)
k2 = np.uint64(0x3333333333333333)
k4 = np.uint64(0x0f0f0f0f0f0f0f0f)
kf = np.uint64(0x0101010101010101)

@njit(uint8(uint64))
def popcount(n):
    n = n - (n >> 1) & k1
    n = (n & k2) + ((n >> 2) & k2)
    n = (n + (n >> 4)) & k4
    n = (n * kf) >> 56
    return n


@vectorize([uint8(uint64)], nopython=True)
def vectorized_popcount(n):
    n = n - (n >> 1) & k1
    n = (n & k2) + ((n >> 2) & k2)
    n = (n + (n >> 4)) & k4
    n = (n * kf) >> 56
    return n


# popcount_lookup_table = np.array([np.sum(np.unpackbits(np.frombuffer(np.uint16(j), np.uint8))) for j in range(2**16)], dtype=np.uint8)
#
# @vectorize([uint8(uint64)], nopython=True)
# def popcount_lookup_table(n):
#     return np.sum(popcount_lookup_table[np.frombuffer(np.array(n),np.uint16)])
#
#
# #I'm pretty sure that np.unpackbits isn't supported by Numba and thus this can't be JIT compiled
# def popcount_unpack_bits(n):
#     return np.sum(np.unpackbits(np.frombuffer(n, dtype=np.uint8)))

@nb.vectorize([nb.uint8(nb.uint8)], nopython=True)
def square_mirror(square):
    """Mirrors the square vertically."""
    return square ^ 0x38

@njit(uint8(uint8))
def square_file(square):
    return square & 7

@njit(uint8(uint8))
def square_rank(square):
    return square >> 3

@njit(uint64(uint64))
def shift_down(b):
    return b >> 8

@njit#(uint64(uint64))
def shift_up(b):
    return b << 8

# @njit#(uint64(uint64))
# def shift_right(b):
#     return b << 1
#
# @njit#(uint64(uint64))
# def shift_left(b):
#     return b >> 1


NOT_BB_FILE_A = ~BB_FILE_A
NOT_BB_FILE_H = ~BB_FILE_H

@njit(nb.uint64(nb.uint64))
def shift_right(b):
    return (b << 1) & NOT_BB_FILE_A

@njit(nb.uint64(nb.uint64))
def shift_left(b):
    return (b >> 1) & NOT_BB_FILE_H


@njit#
def any(iterable):
    for _ in iterable:
        return True
    return False




@njit#(Move.class_type.instance_type(uint8, uint8))
def create_move(from_square, to_square):
    """
    For use when not using promotions
    """
    return Move(from_square, to_square, np.uint8(0))


@njit#(uint8(BoardState.class_type.instance_type, uint8))
def piece_type_at(board_state, square):
    """
    Gets the piece type at the given square.
    """
    mask = BB_SQUARES[square]

    if not board_state.occupied & mask:
        return 0
    elif board_state.pawns & mask:
        return PAWN
    elif board_state.knights & mask:
        return KNIGHT
    elif board_state.bishops & mask:
        return BISHOP
    elif board_state.rooks & mask:
        return ROOK
    elif board_state.queens & mask:
        return QUEEN
    else:
        return KING


@njit#(uint8(BoardState.class_type.instance_type, uint8))
def _remove_piece_at(board_state, square):
    piece_type = piece_type_at(board_state, square)
    mask = BB_SQUARES[square]

    if piece_type == PAWN:
        board_state.pawns ^= mask
    elif piece_type == KNIGHT:
        board_state.knights ^= mask
    elif piece_type == BISHOP:
        board_state.bishops ^= mask
    elif piece_type == ROOK:
        board_state.rooks ^= mask
    elif piece_type == QUEEN:
        board_state.queens ^= mask
    elif piece_type == KING:
        board_state.kings ^= mask
    else:
        return 0

    board_state.occupied ^= mask
    board_state.occupied_w &= ~mask
    board_state.occupied_b &= ~mask

    return piece_type



@njit#(void(BoardState.class_type.instance_type, uint8, uint8, boolean))
def _set_piece_at(board_state, square, piece_type, color):
    _remove_piece_at(board_state, square)

    mask = BB_SQUARES[square]

    if piece_type == PAWN:
        board_state.pawns |= mask
    elif piece_type == KNIGHT:
        board_state.knights |= mask
    elif piece_type == BISHOP:
        board_state.bishops |= mask
    elif piece_type == ROOK:
        board_state.rooks |= mask
    elif piece_type == QUEEN:
        board_state.queens |= mask
    elif piece_type == KING:
        board_state.kings |= mask

    board_state.occupied ^= mask

    if color == TURN_WHITE:
        board_state.occupied_w ^= mask
    else:
        board_state.occupied_b ^= mask


def piece_at(board_state, square):
    piece_type = piece_type_at(board_state, square)
    if piece_type:
        return chess.Piece(int(piece_type), bool(np.uint64(board_state.occupied_w) & BB_SQUARES[square]))
    return None


def create_board_state_from_fen(fen):
    temp_board = chess.Board(fen)

    return BoardState(np.uint64(temp_board.pawns),
                      np.uint64(temp_board.knights),
                      np.uint64(temp_board.bishops),
                      np.uint64(temp_board.rooks),
                      np.uint64(temp_board.queens),
                      np.uint64(temp_board.kings),
                      np.uint64(temp_board.occupied_co[chess.WHITE]),
                      np.uint64(temp_board.occupied_co[chess.BLACK]),
                      np.uint64(temp_board.occupied),
                      np.bool_(temp_board.turn),
                      np.uint64(temp_board.castling_rights),
                      None if temp_board.ep_square is None else np.uint8(temp_board.ep_square),
                      np.uint8(temp_board.halfmove_clock),
                      np.uint64(zobrist_hash(temp_board)))



@njit#(boolean(BoardState.class_type.instance_type, Move.class_type.instance_type))
def is_zeroing(board_state, move):
    """
    Checks if the given pseudo-legal move is a capture or pawn move.
    """
    if board_state.turn:
        return np.bool_(
            BB_SQUARES[move.from_square] & board_state.pawns or BB_SQUARES[move.to_square] & board_state.occupied_b)
    return np.bool_(
        BB_SQUARES[move.from_square] & board_state.pawns or BB_SQUARES[move.to_square] & board_state.occupied_w)


# This should likely not be used/needed long term
@njit#(Move.class_type.instance_type(BoardState.class_type.instance_type, Move.class_type.instance_type))
def _to_chess960(board_state, move):
    if move.from_square == E1 and board_state.kings & BB_E1:
        if move.to_square == G1 and not board_state.rooks & BB_G1:
            return create_move(E1, H1)
        elif move.to_square == C1 and not board_state.rooks & BB_C1:
            return create_move(E1, A1)
    elif move.from_square == E8 and board_state.kings & BB_E8:
        if move.to_square == G8 and not board_state.rooks & BB_G8:
            return create_move(E8, H8)
        elif move.to_square == C8 and not board_state.rooks & BB_C8:
            return create_move(E8, A8)
    return move


@njit#(uint64(BoardState.class_type.instance_type, Move.class_type.instance_type))
def push_with_hash_update(board_state, move):
    # self.hash_stack.append(board_state.cur_hash)
    #
    # # PRETTY SURE THAT I DON'T NEED THIS STACK
    # # self.stack.append(chess._BoardState(self))
    #
    # #Might not even need this stack
    # self.move_stack.append(move)
    #



    # THIS LIKELY SHOULD BE DONE BETTER
    move = _to_chess960(board_state, move)

    # Reset ep square.
    ep_square = board_state.ep_square
    board_state.ep_square = None

    # reset the ep square in the hash
    if not ep_square is None:

        # THIS IS ONLY A TEMPORARY WORKAROUND
        temp_ep_square = np.uint8(ep_square)

        if board_state.turn == True:
            ep_mask = shift_down(BB_SQUARES[temp_ep_square])
            if (shift_left(ep_mask) | shift_right(ep_mask)) & board_state.pawns & board_state.occupied_w:
                board_state.cur_hash ^= RANDOM_ARRAY[772 + square_file(temp_ep_square)]
        else:
            ep_mask = shift_up(BB_SQUARES[temp_ep_square])
            if (shift_left(ep_mask) | shift_right(ep_mask)) & board_state.pawns & board_state.occupied_b:
                board_state.cur_hash ^= RANDOM_ARRAY[772 + square_file(temp_ep_square)]

    # Increment move counters.
    board_state.halfmove_clock += 1
    if board_state.turn == TURN_BLACK:
        pivot = 0
    else:
        pivot = 1


    # Zero the half move clock.
    if is_zeroing(board_state, move):
        board_state.halfmove_clock = 0

    from_bb = BB_SQUARES[move.from_square]
    to_bb = BB_SQUARES[move.to_square]

    piece_type = _remove_piece_at(board_state, move.from_square)

    # Remove the piece that's being moved from the hash
    board_state.cur_hash ^= RANDOM_ARRAY[((piece_type - 1) * 2 + pivot) * 64 + move.from_square]

    capture_square = move.to_square

    captured_piece_type = piece_type_at(board_state, capture_square)

    castle_deltas = board_state.castling_rights

    board_state.castling_rights = board_state.castling_rights & ~to_bb & ~from_bb

    castle_deltas ^= board_state.castling_rights
    # This could likely be sped up by fancier bit twiddling
    if castle_deltas:
        if castle_deltas & BB_A1:
            board_state.cur_hash ^= RANDOM_ARRAY[768 + 1]
        if castle_deltas & BB_H1:
            board_state.cur_hash ^= RANDOM_ARRAY[768]
        if castle_deltas & BB_A8:
            board_state.cur_hash ^= RANDOM_ARRAY[768 + 3]
        if castle_deltas & BB_H8:
            board_state.cur_hash ^= RANDOM_ARRAY[768 + 2]


    if piece_type == KING:
        castle_deltas = board_state.castling_rights
        if board_state.turn == TURN_WHITE:
            board_state.castling_rights &= ~BB_RANK_1
            castle_deltas ^= board_state.castling_rights
            if castle_deltas:
                if castle_deltas & BB_A1:
                    board_state.cur_hash ^= RANDOM_ARRAY[768 + 1]
                if castle_deltas & BB_H1:
                    board_state.cur_hash ^= RANDOM_ARRAY[768]
        else:
            board_state.castling_rights &= ~BB_RANK_8
            castle_deltas ^= board_state.castling_rights
            if castle_deltas:
                if castle_deltas & BB_A8:
                    board_state.cur_hash ^= RANDOM_ARRAY[768 + 3]
                if castle_deltas & BB_H8:
                    board_state.cur_hash ^= RANDOM_ARRAY[768 + 2]


    if piece_type == PAWN:
        if move.to_square >= move.from_square:
            diff = move.to_square - move.from_square
            if diff == 16 and square_rank(move.from_square) == 1:
                board_state.ep_square = move.from_square + 8
            elif not ep_square is None:
                if move.to_square == ep_square and diff in [7, 9] and not captured_piece_type:
                    # Remove pawns captured en passant.
                    if board_state.turn == TURN_WHITE:
                        capture_square = ep_square - 8
                    else:
                        capture_square = ep_square + 8

                    captured_piece_type = _remove_piece_at(board_state, capture_square)

                    # Remove the captured pawn from the hash
                    board_state.cur_hash ^= RANDOM_ARRAY[((pivot + 1) % 2) * 64 + capture_square]
        else:
            diff = move.from_square - move.to_square
            if diff == 16 and square_rank(move.from_square) == 6:
                board_state.ep_square = move.from_square - 8
            elif not ep_square is None:
                if move.to_square == ep_square and diff in [7, 9] and not captured_piece_type:
                    # Remove pawns captured en passant.
                    if board_state.turn == TURN_WHITE:
                        capture_square = ep_square - 8
                    else:
                        capture_square = ep_square + 8

                    captured_piece_type = _remove_piece_at(board_state, capture_square)

                    # Remove the captured pawn from the hash
                    board_state.cur_hash ^= RANDOM_ARRAY[((pivot + 1) % 2) * 64 + capture_square]



    # Promotion.
    if move.promotion != 0:
        piece_type = move.promotion

    # Castling.
    if board_state.turn:
        castling = piece_type == KING and board_state.occupied_w & to_bb
    else:
        castling = piece_type == KING and board_state.occupied_b & to_bb
    if castling:
        _remove_piece_at(board_state, move.from_square)
        _remove_piece_at(board_state, move.to_square)


        if square_file(move.to_square) < square_file(move.from_square):
            if board_state.turn == TURN_WHITE:
                _set_piece_at(board_state, C1, KING, board_state.turn)
                _set_piece_at(board_state, D1, ROOK, board_state.turn)

                board_state.cur_hash ^= RANDOM_ARRAY[((ROOK - 1) * 2 + pivot) * 64 + A1] ^ \
                                        RANDOM_ARRAY[((KING - 1) * 2 + pivot) * 64 + C1] ^ \
                                        RANDOM_ARRAY[((ROOK - 1) * 2 + pivot) * 64 + D1]
            else:
                _set_piece_at(board_state, C8, KING, board_state.turn)
                _set_piece_at(board_state, D8, ROOK, board_state.turn)

                board_state.cur_hash ^= RANDOM_ARRAY[((ROOK - 1) * 2 + pivot) * 64 + A8] ^ \
                                        RANDOM_ARRAY[((KING - 1) * 2 + pivot) * 64 + C8] ^ \
                                        RANDOM_ARRAY[((ROOK - 1) * 2 + pivot) * 64 + D8]
        else:
            if board_state.turn == TURN_WHITE:
                _set_piece_at(board_state, G1, KING, board_state.turn)
                _set_piece_at(board_state, F1, ROOK, board_state.turn)
                board_state.cur_hash ^= RANDOM_ARRAY[((ROOK - 1) * 2 + pivot) * 64 + H1] ^ \
                                        RANDOM_ARRAY[((KING - 1) * 2 + pivot) * 64 + G1] ^ \
                                        RANDOM_ARRAY[((ROOK - 1) * 2 + pivot) * 64 + F1]
            else:
                _set_piece_at(board_state, G8, KING, board_state.turn)
                _set_piece_at(board_state, F8, ROOK, board_state.turn)

                board_state.cur_hash ^= RANDOM_ARRAY[((ROOK - 1) * 2 + pivot) * 64 + H8] ^ \
                                        RANDOM_ARRAY[((KING - 1) * 2 + pivot) * 64 + G8] ^ \
                                        RANDOM_ARRAY[((ROOK - 1) * 2 + pivot) * 64 + F8]

    # Put piece on target square.
    if not castling and piece_type != 0:
        _set_piece_at(board_state, move.to_square, piece_type, board_state.turn)

        # Put the moving piece in the new location in the hash
        board_state.cur_hash ^= RANDOM_ARRAY[((piece_type - 1) * 2 + pivot) * 64 + move.to_square]

        if captured_piece_type:
            if capture_square == move.to_square:
                board_state.cur_hash ^= RANDOM_ARRAY[
                    ((captured_piece_type - 1) * 2 + (pivot + 1) % 2) * 64 + move.to_square]


    # Swap turn.
    board_state.turn = not board_state.turn

    # set the ep square in the hash
    if not board_state.ep_square is None:
        # This is a temporary work around
        temp_ep_square = np.uint8(board_state.ep_square)
        if board_state.turn == TURN_WHITE:
            ep_mask = shift_down(BB_SQUARES[temp_ep_square])
            if (shift_left(ep_mask) | shift_right(ep_mask)) & board_state.pawns & board_state.occupied_w:
                board_state.cur_hash ^= RANDOM_ARRAY[772 + square_file(temp_ep_square)]
        else:
            ep_mask = shift_up(BB_SQUARES[temp_ep_square])
            if (shift_left(ep_mask) | shift_right(ep_mask)) & board_state.pawns & board_state.occupied_b:
                board_state.cur_hash ^= RANDOM_ARRAY[772 + square_file(temp_ep_square)]

    board_state.cur_hash ^= RANDOM_ARRAY[780]
    return board_state.cur_hash


@njit#(BoardState.class_type.instance_type(BoardState.class_type.instance_type, Move.class_type.instance_type))
def copy_push(board_state, move):
    to_push_move = copy_board_state(board_state)
    push_with_hash_update(to_push_move, move)
    return to_push_move


# # WHY AM I TAKING OCCUPIED IN AS A PARAM INSTEAD OF JUST DOING IT ON board_state.occupied####################
# # SHOULD VERY LIKELY DO THIS NOT ALL AT ONCE, INSTEAD ONE AT A TIME BECAUSE IF ONE ATTACKER EXISTS THEN IT'S CHECK (for seeing if board is in check)
# @njit#(uint64(BoardState.class_type.instance_type, boolean, uint8, uint64))
# def _attackers_mask(board_state, color, square, occupied):
#     queens_and_rooks = board_state.queens | board_state.rooks
#     queens_and_bishops = board_state.queens | board_state.bishops
#
#     if color == TURN_WHITE:
#         color_index = BLACK
#     else:
#         color_index = WHITE
#     attackers = (
#         (BB_KING_ATTACKS[square] & board_state.kings) |
#         (BB_KNIGHT_ATTACKS[square] & board_state.knights) |
#         (RANK_ATTACK_ARRAY[square][khash_get(RANK_ATTACK_INDEX_LOOKUP_TABLE, BB_RANK_MASKS[square] & occupied, 0)] & queens_and_rooks) |
#         (FILE_ATTACK_ARRAY[square][khash_get(FILE_ATTACK_INDEX_LOOKUP_TABLE, BB_FILE_MASKS[square] & occupied, 0)] & queens_and_rooks) |
#         (DIAG_ATTACK_ARRAY[square][khash_get(DIAG_ATTACK_INDEX_LOOKUP_TABLE, BB_DIAG_MASKS[square] & occupied, 0)] & queens_and_bishops) |
#         (BB_PAWN_ATTACKS[color_index][square] & board_state.pawns))
#
#     if color == TURN_WHITE:
#         return attackers & board_state.occupied_w
#     return attackers & board_state.occupied_b

# WHY AM I TAKING OCCUPIED IN AS A PARAM INSTEAD OF JUST DOING IT ON board_state.occupied####################
# SHOULD VERY LIKELY DO THIS NOT ALL AT ONCE, INSTEAD ONE AT A TIME BECAUSE IF ONE ATTACKER EXISTS THEN IT'S CHECK (for seeing if board is in check)
@njit#(uint64(BoardState.class_type.instance_type, boolean, uint8, uint64))
def _attackers_mask(board_state, color, square, occupied, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS):
    # rank_pieces = OLD_BB_RANK_MASKS[square] & occupied
    # file_pieces = OLD_BB_FILE_MASKS[square] & occupied
    # diag_pieces = OLD_BB_DIAG_MASKS[square] & occupied
    rank_pieces = BB_RANK_MASKS[square] & occupied
    file_pieces = BB_FILE_MASKS[square] & occupied
    diag_pieces = BB_DIAG_MASKS[square] & occupied


    queens_and_rooks = board_state.queens | board_state.rooks
    queens_and_bishops = board_state.queens | board_state.bishops

    if color == TURN_WHITE:
        color_index = BLACK
    else:
        color_index = WHITE
    # print(type(OLD_BB_DIAG_ATTACKS[square][diag_pieces]), type(queens_and_rooks))
    attackers = (
        (BB_KING_ATTACKS[square] & board_state.kings) |
        (BB_KNIGHT_ATTACKS[square] & board_state.knights) |
        (OLD_BB_RANK_ATTACKS[square][rank_pieces] & queens_and_rooks) |
        (OLD_BB_FILE_ATTACKS[square][file_pieces] & queens_and_rooks) |
        (OLD_BB_DIAG_ATTACKS[square][diag_pieces] & queens_and_bishops) |
        # (RANK_ATTACK_ARRAY[square][khash_get(RANK_ATTACK_INDEX_LOOKUP_TABLE, BB_RANK_MASKS[square] & occupied, 0)] & queens_and_rooks) |
        # (FILE_ATTACK_ARRAY[square][khash_get(FILE_ATTACK_INDEX_LOOKUP_TABLE, BB_FILE_MASKS[square] & occupied, 0)] & queens_and_rooks) |
        # (DIAG_ATTACK_ARRAY[square][khash_get(DIAG_ATTACK_INDEX_LOOKUP_TABLE, BB_DIAG_MASKS[square] & occupied, 0)] & queens_and_bishops) |
        (BB_PAWN_ATTACKS[color_index][square] & board_state.pawns))

    if color == TURN_WHITE:
        return attackers & board_state.occupied_w
    return attackers & board_state.occupied_b

@njit#(uint64(BoardState.class_type.instance_type, uint8))
def attacks_mask(board_state, square, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS):
    bb_square = BB_SQUARES[square]

    if bb_square & board_state.pawns:
        if bb_square & board_state.occupied_w:
            return BB_PAWN_ATTACKS[WHITE][square]
        else:
            return BB_PAWN_ATTACKS[BLACK][square]
    elif bb_square & board_state.knights:
        return BB_KNIGHT_ATTACKS[square]
    elif bb_square & board_state.kings:
        return BB_KING_ATTACKS[square]
    else:
        attacks = np.uint64(0)
        if bb_square & board_state.bishops or bb_square & board_state.queens:
            attacks = OLD_BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & board_state.occupied]
            # attacks = DIAG_ATTACK_ARRAY[square][OLD_BB_DIAG_MASKS[square] & board_state.occupied]
                # khash_get(DIAG_ATTACK_INDEX_LOOKUP_TABLE, BB_DIAG_MASKS[square] & board_state.occupied, 0)]
        if bb_square & board_state.rooks or bb_square & board_state.queens:

            attacks |= (OLD_BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & board_state.occupied] |
                        OLD_BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & board_state.occupied])

            # attacks |= (RANK_ATTACK_ARRAY[square][
            #                 khash_get(RANK_ATTACK_INDEX_LOOKUP_TABLE, BB_RANK_MASKS[square] & board_state.occupied,0)] |
            #             FILE_ATTACK_ARRAY[square][
            #                 khash_get(FILE_ATTACK_INDEX_LOOKUP_TABLE, BB_FILE_MASKS[square] & board_state.occupied, 0)])
        return attacks


@njit#((BoardState.class_type.instance_type, uint64, uint64))
def generate_pseudo_legal_ep(board_state, from_mask=BB_ALL, to_mask=BB_ALL):
    if board_state.ep_square is None:
        return

    temp_ep_square = np.uint8(board_state.ep_square)
    if not BB_SQUARES[temp_ep_square] & to_mask:
        return

    if BB_SQUARES[temp_ep_square] & board_state.occupied:
        return
    if board_state.turn:
        capturers = (
            board_state.pawns & board_state.occupied_w & from_mask &
            BB_PAWN_ATTACKS[BLACK][temp_ep_square] & BB_RANKS[4])
    else:
        capturers = (
            board_state.pawns & board_state.occupied_b & from_mask &
            BB_PAWN_ATTACKS[WHITE][temp_ep_square] & BB_RANKS[3])

    for capturer in scan_reversed(capturers):
        yield create_move(capturer, temp_ep_square)

    return



@njit#(uint64(BoardState.class_type.instance_type, uint8))
def _slider_blockers(board_state, king, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS):
    ###################################################### THESE CAN GO IN THE BELOW snipers INITIALIZATION
    rooks_and_queens = board_state.rooks | board_state.queens
    bishops_and_queens = board_state.bishops | board_state.queens
    # print(type(king), king)
    # a = OLD_BB_RANK_ATTACKS[king][np.uint64(0)]
    # b = OLD_BB_FILE_ATTACKS[king][np.uint64(0)]
    # print(type(OLD_BB_RANK_ATTACKS[king][0]),type(OLD_BB_FILE_ATTACKS[king][0]), type(OLD_BB_DIAG_ATTACKS[king][0]), type(rooks_and_queens))
    # king = np.int16(king)
    # print(OLD_BB_RANK_ATTACKS[king][0], type(king))
    # c = OLD_BB_DIAG_ATTACKS[king][np.uint64(0)]
    snipers = ((OLD_BB_RANK_ATTACKS[king][np.uint64(0)] & rooks_and_queens) |
               (OLD_BB_FILE_ATTACKS[king][np.uint64(0)] & rooks_and_queens) |
               (OLD_BB_DIAG_ATTACKS[king][np.uint64(0)] & bishops_and_queens))
    # snipers = ((RANK_ATTACK_ARRAY[king][0] & rooks_and_queens) |
    #            (FILE_ATTACK_ARRAY[king][0] & rooks_and_queens) |
    #            (DIAG_ATTACK_ARRAY[king][0] & bishops_and_queens))

    blockers = 0
    if board_state.turn == TURN_WHITE:
        for sniper in scan_reversed(snipers & board_state.occupied_b):
            b = BB_BETWEEN[king][sniper] & board_state.occupied

            # Add to blockers if exactly one piece in between.
            if b and BB_SQUARES[msb(b)] == b:
                blockers |= b

        return blockers & board_state.occupied_w
    else:
        for sniper in scan_reversed(snipers & board_state.occupied_w):
            b = BB_BETWEEN[king][sniper] & board_state.occupied

            # Add to blockers if exactly one piece in between.
            if b and BB_SQUARES[msb(b)] == b:
                blockers |= b

        return blockers & board_state.occupied_b


@njit#(boolean(BoardState.class_type.instance_type, Move.class_type.instance_type))
def is_castling(board_state, move):
    """
    Checks if the given pseudo-legal move is a castling move.
    """
    if board_state.kings & BB_SQUARES[move.from_square]:
        # THIS IS A TEMPORARY WORKAROUND FOR ALLOWING abs(a-b) where a < b, and a and b are unsigned integers
        from_file = square_file(move.from_square)
        to_file = square_file(move.to_square)
        if from_file > to_file:
            diff = from_file - to_file
        else:
            diff = to_file - from_file

        if board_state.turn == TURN_WHITE:
            return diff > 1 or bool(board_state.rooks & board_state.occupied_w & BB_SQUARES[move.to_square])
        else:
            return diff > 1 or bool(board_state.rooks & board_state.occupied_b & BB_SQUARES[move.to_square])

    return False


@njit#(boolean(BoardState.class_type.instance_type, uint64, uint64))
def _attacked_for_king(board_state, path, occupied, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS):
    for sq in scan_reversed(path):
        if _attackers_mask(board_state, not board_state.turn, sq, occupied, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS):
            return True
    return False


@njit#(uint64(BoardState.class_type.instance_type, uint64, uint8))
def _castling_uncovers_rank_attack(board_state, rook_bb, king_to, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS):
    """
    Test the special case where we castle and our rook shielded us from
    an attack, so castling would be into check.
    """
    rank_pieces = BB_RANK_MASKS[king_to] & (board_state.occupied ^ rook_bb)
    if board_state.turn:
        sliders = (board_state.queens | board_state.rooks) & board_state.occupied_b
    else:
        sliders = (board_state.queens | board_state.rooks) & board_state.occupied_w
    # return RANK_ATTACK_ARRAY[king_to][khash_get(RANK_ATTACK_INDEX_LOOKUP_TABLE, rank_pieces, 0)] & sliders
    # a = OLD_BB_RANK_ATTACKS[rank_pieces]
    # b = OLD_BB_RANK_ATTACKS[king_to]
    # c = b[a]
    return OLD_BB_RANK_ATTACKS[king_to][rank_pieces] & sliders
    # return c & sliders


@njit#(Move.class_type.instance_type(BoardState.class_type.instance_type, uint8, uint8))
def _from_chess960(board_state, from_square, to_square):  # , promotion=None):
    # if not chess960 and drop is None:
    if from_square == E1 and board_state.kings & BB_E1:
        if to_square == H1:
            return create_move(E1, G1)
        elif to_square == A1:
            return create_move(E1, C1)
    elif from_square == E8 and board_state.kings & BB_E8:
        if to_square == H8:
            return create_move(E8, G8)
        elif to_square == A8:
            return create_move(E8, C8)

    # promotion is set to None because this function only gets called when looking for castling moves,
    # which can't have promotions
    return create_move(from_square, to_square)


@njit#((BoardState.class_type.instance_type, uint64, uint64))
def generate_castling_moves(board_state, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS, from_mask=BB_ALL, to_mask=BB_ALL):
    if board_state.turn:
        backrank = BB_RANK_1 #if board_state.turn == TURN_WHITE else BB_RANK_8
        king = board_state.occupied_w & board_state.kings & backrank & from_mask
    else:
        backrank = BB_RANK_8 # if board_state.turn == TURN_WHITE else BB_RANK_8
        king = board_state.occupied_b & board_state.kings & backrank & from_mask

    king = king & -king

    ###################(on second thought this comment might be wrong when doing things like checking for legality of moves) REALLY SHOULD CHANGE THIS STUFF BECAUSE IT'S CHECKING FOR THINGS LIKE IF THERE IS A KING AT ALL ON THE BOARD
    if not king or _attacked_for_king(board_state, king, board_state.occupied, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS):
        return

    bb_c = BB_FILE_C & backrank
    bb_d = BB_FILE_D & backrank
    bb_f = BB_FILE_F & backrank
    bb_g = BB_FILE_G & backrank

    # for candidate in scan_reversed(self.clean_castling_rights() & backrank & to_mask):
    for candidate in scan_reversed(board_state.castling_rights & backrank & to_mask):
        rook = BB_SQUARES[candidate]

        empty_for_rook = np.uint64(0)
        empty_for_king = np.uint64(0)

        if rook < king:
            king_to = msb(bb_c)
            if not rook & bb_d:
                empty_for_rook = BB_BETWEEN[candidate][msb(bb_d)] | bb_d
            if not king & bb_c:
                empty_for_king = BB_BETWEEN[msb(king)][king_to] | bb_c
        else:
            king_to = msb(bb_g)
            if not rook & bb_f:
                empty_for_rook = BB_BETWEEN[candidate][msb(bb_f)] | bb_f
            if not king & bb_g:
                empty_for_king = BB_BETWEEN[msb(king)][king_to] | bb_g

        if not ((board_state.occupied ^ king ^ rook) & (empty_for_king | empty_for_rook) or
                    _attacked_for_king(board_state, empty_for_king, board_state.occupied ^ king, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS) or
                    _castling_uncovers_rank_attack(board_state, rook, king_to, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS)):
            yield _from_chess960(board_state, msb(king), candidate)

    return


@njit#((BoardState.class_type.instance_type, uint64, uint64))
def generate_pseudo_legal_moves(board_state, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS, from_mask=BB_ALL, to_mask=BB_ALL):
    if board_state.turn == TURN_WHITE:
        cur_turn_occupied = board_state.occupied_w
        opponent_occupied = board_state.occupied_b
    else:
        cur_turn_occupied = board_state.occupied_b
        opponent_occupied = board_state.occupied_w

    our_pieces = cur_turn_occupied

    # Generate piece moves.
    non_pawns = our_pieces & ~board_state.pawns & from_mask

    for from_square in scan_reversed(non_pawns):

        moves = attacks_mask(board_state, from_square, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS) & ~our_pieces & to_mask
        for to_square in scan_reversed(moves):
            yield create_move(from_square, to_square)

    # Generate castling moves.
    if from_mask & board_state.kings:
        for move in generate_castling_moves(board_state, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS, from_mask, to_mask):
            yield move

    # The remaining moves are all pawn moves.
    pawns = board_state.pawns & cur_turn_occupied & from_mask
    if not pawns:
        return

    # Generate pawn captures.
    capturers = pawns
    for from_square in scan_reversed(capturers):
        if board_state.turn:
            targets = (
                BB_PAWN_ATTACKS[WHITE][from_square] &
                opponent_occupied & to_mask)
        else:
            targets = (
                BB_PAWN_ATTACKS[BLACK][from_square] &
                opponent_occupied & to_mask)

        for to_square in scan_reversed(targets):
            if square_rank(to_square) in [0, 7]:
                yield Move(from_square, to_square, QUEEN)
                yield Move(from_square, to_square, ROOK)
                yield Move(from_square, to_square, BISHOP)
                yield Move(from_square, to_square, KNIGHT)
            else:
                yield create_move(from_square, to_square)

    # Prepare pawn advance generation.
    if board_state.turn == TURN_WHITE:
        single_moves = pawns << 8 & ~board_state.occupied
        double_moves = single_moves << 8 & ~board_state.occupied & (BB_RANK_3 | BB_RANK_4)
    else:
        single_moves = pawns >> 8 & ~board_state.occupied
        double_moves = single_moves >> 8 & ~board_state.occupied & (BB_RANK_6 | BB_RANK_5)

    single_moves &= to_mask
    double_moves &= to_mask

    # Generate single pawn moves.
    for to_square in scan_reversed(single_moves):
        if board_state.turn == TURN_BLACK:
            from_square = to_square + 8
        else:
            from_square = to_square - 8

        if square_rank(to_square) in [0, 7]:
            yield Move(from_square, to_square, QUEEN)
            yield Move(from_square, to_square, ROOK)
            yield Move(from_square, to_square, BISHOP)
            yield Move(from_square, to_square, KNIGHT)
        else:
            yield create_move(from_square, to_square)



    # Generate double pawn moves.
    for to_square in scan_reversed(double_moves):
        if board_state.turn == TURN_BLACK:
            from_square = to_square + 16
        else:
            from_square = to_square - 16
        yield create_move(from_square, to_square)


    # Generate en passant captures.
    if not board_state.ep_square is None:
        for move in generate_pseudo_legal_ep(board_state, from_mask, to_mask):
            yield move

    return


@njit#((BoardState.class_type.instance_type, uint8, uint64, uint64, uint64))
def _generate_evasions(board_state, king, checkers, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS, from_mask=BB_ALL, to_mask=BB_ALL):
    sliders = checkers & (board_state.bishops | board_state.rooks | board_state.queens)

    attacked = np.uint64(0)
    for checker in scan_reversed(sliders):
        attacked |= BB_RAYS[king][checker] & ~BB_SQUARES[checker]

    if BB_SQUARES[king] & from_mask:
        if board_state.turn:
            for to_square in scan_reversed(BB_KING_ATTACKS[king] & ~board_state.occupied_w & ~attacked & to_mask):
                yield create_move(king, to_square)
        else:
            for to_square in scan_reversed(BB_KING_ATTACKS[king] & ~board_state.occupied_b & ~attacked & to_mask):
                yield create_move(king, to_square)

    checker = msb(checkers)
    if BB_SQUARES[checker] == checkers:
        # Capture or block a single checker.
        target = BB_BETWEEN[king][checker] | checkers


        for move in generate_pseudo_legal_moves(board_state, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS, ~board_state.kings & from_mask, target & to_mask):
            yield move

        # Capture the checking pawn en passant (but avoid yielding duplicate moves).
        if not board_state.ep_square is None:
            temp_ep_square = np.uint8(board_state.ep_square)
            if not BB_SQUARES[temp_ep_square] & target:
                if board_state.turn == TURN_BLACK:
                    last_double = temp_ep_square + 8
                else:
                    last_double = temp_ep_square - 8
                if last_double == checker:
                    for move in generate_pseudo_legal_ep(board_state, from_mask, to_mask):
                        yield move

    return


@njit#(boolean(BoardState.class_type.instance_type, Move.class_type.instance_type))
def is_en_passant(board_state, move):
    """
    Checks if the given pseudo-legal move is an en passant capture.
    """
    if not board_state.ep_square is None:
        if move.to_square >= move.from_square:
            return (board_state.ep_square == move.to_square and
                    bool(board_state.pawns & BB_SQUARES[move.from_square]) and
                    move.to_square - move.from_square in [7, 9] and
                    not board_state.occupied & BB_SQUARES[move.to_square])
        else:
            return (board_state.ep_square == move.to_square and
                    bool(board_state.pawns & BB_SQUARES[move.from_square]) and
                    move.from_square - move.to_square in [7, 9] and
                    not board_state.occupied & BB_SQUARES[move.to_square])

    # Pretty sure this return is needed for compilation
    return False


@njit#(uint64(BoardState.class_type.instance_type, boolean, uint8))
def pin_mask(board_state, color, square, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS):
    if color:
        king = msb(board_state.occupied_w & board_state.kings)
    else:
        king = msb(board_state.occupied_b & board_state.kings)

    square_mask = BB_SQUARES[square]
    # for attacks, sliders in zip((FILE_ATTACK_ARRAY, RANK_ATTACK_ARRAY, DIAG_ATTACK_ARRAY), (
    for attacks, sliders in zip((OLD_BB_FILE_ATTACKS, OLD_BB_RANK_ATTACKS, OLD_BB_DIAG_ATTACKS), (
        board_state.rooks | board_state.queens, board_state.rooks | board_state.queens,
        board_state.bishops | board_state.queens)):
        rays = attacks[king][0]
        if rays & square_mask:
            if color:
                snipers = rays & sliders & board_state.occupied_b
            else:
                snipers = rays & sliders & board_state.occupied_w

            for sniper in scan_reversed(snipers):
                if BB_BETWEEN[sniper][king] & (board_state.occupied | square_mask) == square_mask:
                    return BB_RAYS[king][sniper]

            break

    return BB_ALL


@njit#(boolean(BoardState.class_type.instance_type, uint8, uint8))
def _ep_skewered(board_state, king, capturer, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS):
    """
    Handle the special case where the king would be in check, if the pawn and its capturer disappear from the rank.

    Vertical skewers of the captured pawn are not possible. (Pins on the capturer are not handled here.)
    """

    # only using as workaround, won't do this long term
    temp_ep_square = np.uint8(board_state.ep_square)
    if board_state.turn:
        last_double = temp_ep_square - 8
        occupancy = (board_state.occupied & ~BB_SQUARES[last_double] & ~BB_SQUARES[capturer] | BB_SQUARES[temp_ep_square])

        # Horizontal attack on the fifth or fourth rank.
        horizontal_attackers = board_state.occupied_b & (board_state.rooks | board_state.queens)
        if OLD_BB_RANK_ATTACKS[king][BB_RANK_MASKS[king] & occupancy] & horizontal_attackers:
            return True
        # if RANK_ATTACK_ARRAY[king][khash_get(RANK_ATTACK_INDEX_LOOKUP_TABLE, BB_RANK_MASKS[king] & occupancy, 0)] & horizontal_attackers:
        #     return True

    else:
        last_double = temp_ep_square + 8
        occupancy = (board_state.occupied & ~BB_SQUARES[last_double] & ~BB_SQUARES[capturer] | BB_SQUARES[temp_ep_square])

        # Horizontal attack on the fifth or fourth rank.
        horizontal_attackers = board_state.occupied_w & (board_state.rooks | board_state.queens)
        if OLD_BB_RANK_ATTACKS[king][BB_RANK_MASKS[king] & occupancy] & horizontal_attackers:
            return True
        # if RANK_ATTACK_ARRAY[king][
        #     khash_get(RANK_ATTACK_INDEX_LOOKUP_TABLE, BB_RANK_MASKS[king] & occupancy, 0)] & horizontal_attackers:
        #     return True

    return False


@njit#(boolean(BoardState.class_type.instance_type))
def is_in_check(board_state):
    if board_state.turn == TURN_WHITE:
        king = msb(board_state.occupied_w & board_state.kings)
        return bool(_attackers_mask(board_state, TURN_BLACK, king, board_state.occupied))
    else:
        king = msb(board_state.occupied_b & board_state.kings)
        return bool(_attackers_mask(board_state, TURN_WHITE, king, board_state.occupied))


# @njit#(boolean(BoardState.class_type.instance_type, uint8, uint64, Move.class_type.instance_type))
# @jit
def _is_safe(board_state, king, blockers, move, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS):
    print(type(move.from_square), type(king))
    if move.from_square == king:
        if is_castling(board_state, move):
            print("DLKJSLDKFJSDLKJFDSDDDDDDDDDDDDDDDDDDDD")
            return True
        else:
            return not bool(_attackers_mask(board_state, not board_state.turn, move.to_square, board_state.occupied, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS))

    elif is_en_passant(board_state, move):  # THIS SHOULD REALLY BE DONE WITHIN THIS FUNCTION
        pin = pin_mask(board_state, board_state.turn, move.from_square, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS)
        skewered = _ep_skewered(board_state, king, move.from_square, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS)
        return (pin & BB_SQUARES[move.to_square] and
                not skewered)
    else:
        print("AAAAAAAAAAAA")
        print(type(BB_RAYS[move.from_square][move.to_square]))
        print(type(BB_SQUARES[king]))
        return (not blockers & BB_SQUARES[move.from_square] or
                BB_RAYS[move.from_square][move.to_square] & BB_SQUARES[king])



# @njit#((BoardState.class_type.instance_type, uint64, uint64), nogil=True)
def generate_legal_moves(board_state, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS, from_mask=BB_ALL, to_mask=BB_ALL):

    if board_state.turn == TURN_WHITE:
        king = msb(board_state.kings & board_state.occupied_w)
    else:
        king = msb(board_state.kings & board_state.occupied_b)
    blockers = _slider_blockers(board_state, king, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS)
    checkers = _attackers_mask(board_state, not board_state.turn, king, board_state.occupied, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS)
    #If in check
    if checkers:#If no moves are found it needs to be passed along that the board is in check, so it doesn't need to be computed again directly afterwards
        # print("OH NOIODIJFOSJFKLDSJ")
        for move in _generate_evasions(board_state, king, checkers,OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS,  from_mask, to_mask):
            if _is_safe(board_state, king, blockers, move, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS):
                yield move
    else:#If no moves are found it needs to be passed along that the board is not in check, so it doesn't need to be computed again directly afterwards
        for move in generate_pseudo_legal_moves(board_state, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS, from_mask, to_mask):
            if _is_safe(board_state, king, blockers, move, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS):
                yield move



# @njit(nogil=True)#((BoardState.class_type.instance_type, uint64, uint64), nogil=True)
# def create_legal_move_struct(board_state, from_mask=BB_ALL, to_mask=BB_ALL):
#     """
#     Creates a NumPy structued array of the legal moves for the given BoardState object.  If no moves exist,
#     it will return None.
#     """
#
#     move_counter = 0
#     for move in generate_legal_moves(board_state, from_mask, to_mask):
#         if move_counter == 0:
#             move_array = np.zeros_like(EMPTY_MOVE_ARRAY)
#         move_array[move_counter].to_square = move.to_square
#         move_array[move_counter].from_square = move.from_square
#         move_array[move_counter].promotion = move.promotion
#         move_counter += 1
#
#     if move_counter == 0:
#         return None
#     else:
#         return move_array[:move_counter]



@njit(nogil=True)#((BoardState.class_type.instance_type, uint64, uint64), nogil=True)
def create_legal_move_struct_without_move(board_state, move_not_to_make, from_mask=BB_ALL, to_mask=BB_ALL):
    """
    Does the same as create_legal_move_struct, except that it prevents a specified move from being created.
    This function was created differently than create_legal_move_struct so that it could prevent some calls
    to _is_safe.
    """
    if board_state.turn == TURN_WHITE:
        king = msb(board_state.kings & board_state.occupied_w)
    else:
        king = msb(board_state.kings & board_state.occupied_b)
    blockers = _slider_blockers(board_state, king)
    checkers = _attackers_mask(board_state, not board_state.turn, king, board_state.occupied)

    move_counter = 0

    # If in check
    if checkers:  #If no moves are found it needs to be passed along that the board is in check, so it doesn't need to be computed again directly afterwards
        for move in _generate_evasions(board_state, king, checkers, from_mask, to_mask):
            if not move_objects_equal(move, move_not_to_make) and _is_safe(board_state, king, blockers, move):
                if move_counter == 0:
                    move_array = np.zeros_like(EMPTY_MOVE_ARRAY)
                move_array[move_counter].to_square = move.to_square
                move_array[move_counter].from_square = move.from_square
                move_array[move_counter].promotion = move.promotion
                move_counter += 1
    else:  #If no moves are found it needs to be passed along that the board is not in check, so it doesn't need to be computed again directly afterwards
        for move in generate_pseudo_legal_moves(board_state, from_mask, to_mask):
            if not move_objects_equal(move, move_not_to_make) and _is_safe(board_state, king, blockers, move):
                if move_counter == 0:
                    move_array = np.zeros_like(EMPTY_MOVE_ARRAY)
                move_array[move_counter].to_square = move.to_square
                move_array[move_counter].from_square = move.from_square
                move_array[move_counter].promotion = move.promotion
                move_counter += 1

    if move_counter == 0:
        return None
    else:
        return move_array[:move_counter]


# @njit#(boolean(BoardState.class_type.instance_type), nogil=True)
# def has_legal_move(board_state):
#     """
#     Checks if there exists a legal move
#     """
#     return any(generate_legal_moves(board_state, BB_ALL, BB_ALL))



@njit#(boolean(BoardState.class_type.instance_type, Move.class_type.instance_type))
def is_into_check(board_state, move, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS):
    """
    Checks if the given move would leave the king in check or put it into
    check. The move must be at least pseudo legal.  This function was adapted from the Python-Chess version of it.
    """

    if board_state.turn == TURN_WHITE:
        king = msb(board_state.occupied_w & board_state.kings)
    else:
        king = msb(board_state.occupied_b & board_state.kings)

    if king is None:############################I THINK I CAN REMOVE THIS###################3
        return False

    checkers = _attackers_mask(board_state, not board_state.turn, king, board_state.occupied)
    if checkers:
        # If already in check, look if it is an evasion.
        for temp_move in _generate_evasions(board_state, king, checkers, BB_SQUARES[move.from_square], BB_SQUARES[move.to_square]):
            if move_objects_equal(move, temp_move):
                return True

    return not _is_safe(board_state, king, _slider_blockers(board_state, king), move, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS)




@njit#(boolean(BoardState.class_type.instance_type, Move.class_type.instance_type))
def is_pseudo_legal(board_state, move, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS):
    """
    Checks if a given move is pseudo legal.  This function was adapted from the Python-Chess version.
    """
    # Source square must not be vacant.
    piece = piece_type_at(board_state, move.from_square)
    if not piece:
        return False

    # Get square masks.
    from_mask = BB_SQUARES[move.from_square]
    to_mask = BB_SQUARES[move.to_square]

    # Check turn.
    if board_state.turn == TURN_WHITE:
        if not board_state.occupied_w & from_mask:
            return False
    else:
        if not board_state.occupied_b & from_mask:
            return False

    # Only pawns can promote and only on the backrank.
    if move.promotion:
        if piece != PAWN:
            return False

        if board_state.turn == TURN_WHITE and square_rank(move.to_square) != 7:
            return False
        elif board_state.turn == TURN_BLACK and square_rank(move.to_square) != 0:
            return False

    # Handle castling.
    if piece == KING:
        ####This should not be taking in BB_ALL and instead take in from_mask and to_mask,
        # but I haven't tested or looked into if that correct yet, and I want to get this commit out so for now
        # this slightly slower version will work wine
        for temp_move in generate_castling_moves(board_state, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS, BB_ALL, BB_ALL):
            if move_objects_equal(move, temp_move):
                return True

    # Destination square can not be occupied.
    if board_state.turn == TURN_WHITE:
        if board_state.occupied_w & to_mask:
            return False
    else:
        if board_state.occupied_b & to_mask:
            return False

    # Handle pawn moves.
    if piece == PAWN:
        for temp_move in generate_pseudo_legal_moves(board_state, from_mask, to_mask):
            if move_objects_equal(move, temp_move):
                return True
        return False

    # Handle all other pieces.
    return bool(attacks_mask(board_state, move.from_square) & to_mask)





@njit#(boolean(BoardState.class_type.instance_type, Move.class_type.instance_type))
def is_legal_move(board_state, move, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS):
    """
    Checks if a given move is legal for the given board.
    """
    return is_pseudo_legal(board_state, move, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS) and not is_into_check(board_state, move, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS)



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

    board_state = create_board_state_from_fen('8/8/8/8/8/8/8/8 w - - 0 1')
    for square in SQUARES[:len(SQUARES)//2]:
        for piece in (KNIGHT, QUEEN):
            _set_piece_at(board_state, square, piece, TURN_WHITE)
            for move in generate_legal_moves(board_state,BB_ALL, BB_ALL):
                if possible_moves.get((move.from_square, move.to_square)) is None:
                    possible_moves[move.from_square, move.to_square] = len(possible_moves)
            _remove_piece_at(board_state, square)

    switch_square_fn = lambda x: 8 * (7 - (x >> 3)) + (x & 7)

    total_possible_moves = len(possible_moves)*2 - 1

    for (from_square, to_square), move_num in list(possible_moves.items()):
        possible_moves[switch_square_fn(from_square), switch_square_fn(to_square)] = total_possible_moves - move_num

    return possible_moves


# @njit#(uint64(BoardState.class_type.instance_type, uint8))
def jitted_perft_test(board_state, depth, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS):
    if depth == 0:
        return np.uint64(1)

    num_nodes = np.uint64(0)

    for move in generate_legal_moves(board_state, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS, BB_ALL, BB_ALL):
        # print(chess.Move(move.from_square, move.to_square, move.promotion))
        num_nodes += jitted_perft_test(copy_push(board_state, move), depth - 1, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS)

    return num_nodes


if __name__ == "__main__":
    starting_time = time.time()
    perft_results = jitted_perft_test(
        # flipped_board_state(create_board_state_from_fen('rnbqkbnr/pppppppp/8/8/8/7N/PPPPPPPP/RNBQKB1R b KQkq - 1 1')),
        # create_board_state_from_fen('rnbqkb1r/pppppppp/7n/8/8/7N/PPPPPPPP/RNBQKB1R w KQkq - 2 2'),
        create_board_state_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),#"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        1,
        OLD_BB_RANK_ATTACKS,
        OLD_BB_FILE_ATTACKS,
        OLD_BB_DIAG_ATTACKS,
    )
    print("Depth 4 perft test results:", perft_results, "in time", time.time()-starting_time, "\n")
# jitted_perft_test()