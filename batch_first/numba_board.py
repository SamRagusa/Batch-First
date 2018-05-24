from chess.polyglot import zobrist_hash

from . import *
# from engine_constants import *


# import llvmlite.binding as llvm
# llvm.set_option('', '--debug-only=loop-vectorize')





numpy_node_info_dtype = np.dtype([("pawns", np.uint64),
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
                                  ("hash", np.uint64),
                                  ("terminated", np.bool_),
                                  ("separator", np.float32),
                                  ("depth", np.uint8),
                                  ("best_value", np.float32),
                                  ("unexplored_moves", np.uint8, (MAX_MOVES_LOOKED_AT, 3)),
                                  ("unexplored_move_scores", np.float32, (MAX_MOVES_LOOKED_AT)),
                                  ('prev_move', np.uint8, (3)),
                                  ("next_move_index", np.uint8),
                                  ("children_left", np.uint8)])

numba_node_info_type = nb.from_dtype(numpy_node_info_dtype)


def create_node_info_from_fen(fen, depth, separator):
    return create_node_info_from_python_chess_board(chess.Board(fen), depth, separator)


def create_node_info_from_python_chess_board(board, depth=255, separator=0):
    return np.array([(board.pawns,
                      board.knights,
                      board.bishops,
                      board.rooks,
                      board.queens,
                      board.kings,
                      board.occupied_co[chess.WHITE],
                      board.occupied_co[chess.BLACK],
                      board.occupied,
                      board.turn,
                      board.castling_rights,
                      board.ep_square if not board.ep_square is None else 0,
                      board.halfmove_clock,
                      zobrist_hash(board),
                      False,                                                                  #terminated
                      separator,
                      depth,
                      MIN_FLOAT32_VAL,                                                        #best_value
                      np.full([MAX_MOVES_LOOKED_AT, 3], 255, dtype=np.uint8),                 #unexplored moves
                      np.full([MAX_MOVES_LOOKED_AT], MIN_FLOAT32_VAL, dtype=np.float32),   #unexplored move scores
                      np.full([3], 255, dtype=np.uint8), #The move made to reach the position this board represents
                      0,           #next_move_index  (the index in the stored moves where the next move to make is)
                      0)                   #children_left (the number of children which have yet to returne a value, or be created)
                     ],dtype=numpy_node_info_dtype)




flip_vert_const_1 = np.uint64(0x00FF00FF00FF00FF)
flip_vert_const_2 = np.uint64(0x0000FFFF0000FFFF)

@nb.vectorize([nb.uint64(nb.uint64)])
def vectorized_flip_vertically(bb):
    bb = ((bb >>  8) & flip_vert_const_1) | ((bb & flip_vert_const_1) <<  8)
    bb = ((bb >> 16) & flip_vert_const_2) | ((bb & flip_vert_const_2) << 16)
    bb = ( bb >> 32) | ( bb << 32)
    return bb



# Obviously needs to be refactored
@njit(nb.uint8(nb.uint64))
def msb(n):
    r = 0
    n = n >> 1
    while n:
        r += 1
        n = n >> 1
    return r


# Obviously needs to be refactored
@nb.vectorize([nb.uint8(nb.uint64)],nopython=True)
def vectorized_msb(n):
    r = 0
    n = n >> 1
    while n:  # >>= 1:
        r += 1
        n = n >> 1
    return r




# Obviously needs to be refactored
@njit
def scan_reversed(bb):
    for index in range(BB_SQUARES.shape[0]):
        if bb & BB_SQUARES[index]:
            yield np.uint8(index)
            bb ^= BB_SQUARES[index]
    return



popcount_const_1 = np.uint64(0x5555555555555555)
popcount_const_2 = np.uint64(0x3333333333333333)
popcount_const_3 = np.uint64(0x0f0f0f0f0f0f0f0f)
popcount_const_4 = np.uint64(0x0101010101010101)

@njit(nb.uint8(nb.uint64))
def popcount(n):
    n = n - (n >> 1) & popcount_const_1
    n = (n & popcount_const_2) + ((n >> 2) & popcount_const_2)
    n = (n + (n >> 4)) & popcount_const_3
    n = (n * popcount_const_4) >> 56
    return n


@nb.vectorize([nb.uint8(nb.uint64)], nopython=True)
def vectorized_popcount(n):
    n = n - (n >> 1) & popcount_const_1
    n = (n & popcount_const_2) + ((n >> 2) & popcount_const_2)
    n = (n + (n >> 4)) & popcount_const_3
    n = (n * popcount_const_4) >> 56
    return n


@njit(nb.uint8(nb.uint8))
def square_file(square):
    return square & 7

@njit(nb.uint8(nb.uint8))
def square_rank(square):
    return square >> 3

@njit(nb.uint64(nb.uint64))
def shift_down(b):
    return b >> 8

@njit(nb.uint64(nb.uint64))
def shift_up(b):
    return b << 8

@njit(nb.uint64(nb.uint64))
def shift_right(b):
    return (b << 1) & ~BB_FILE_A

@njit(nb.uint64(nb.uint64))
def shift_left(b):
    return (b >> 1) & ~BB_FILE_H

@nb.vectorize([nb.uint8(nb.uint8)], nopython=True)
def vectorized_square_mirror(square):
    """Mirrors the square vertically."""
    return square ^ 0x38

@njit(nb.uint8(nb.uint8))
def square_mirror(square):
    """Mirrors the square vertically."""
    return square

@njit
def any(iterable):
    for _ in iterable:
        return True
    return False



@njit
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


@njit
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



@njit
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


@njit
def scalar_is_zeroing(board_state, move_from_square, move_to_square):
    """
    Checks if the given pseudo-legal move is a capture or pawn move.
    """
    if board_state.turn:
        return np.bool_(
            BB_SQUARES[move_from_square] & board_state.pawns or BB_SQUARES[move_to_square] & board_state.occupied_b)
    return np.bool_(
        BB_SQUARES[move_from_square] & board_state.pawns or BB_SQUARES[move_to_square] & board_state.occupied_w)



@njit
def _to_chess960_tuple(board_state, move):
    if move[0] == E1 and board_state.kings & BB_E1:
        if move[1] == G1 and not board_state.rooks & BB_G1:
            return E1, H1, 0
        elif move[1] == C1 and not board_state.rooks & BB_C1:
            return E1, A1, 0
    elif move[0] == E8 and board_state.kings & BB_E8:
        if move[1] == G8 and not board_state.rooks & BB_G8:
            return E8, H8, 0
        elif move[1] == C8 and not board_state.rooks & BB_C8:
            return E8, A8, 0
    return move[0], move[1], move[2]


@njit
def push_moves(struct_array, move_array):
    """
    Pushes the given moves for the given boards (makes the moves), while doing this it also incrementally updates
    the structs internally stored Zobrist hassh.


    :param struct_array: An ndarray with dtype numpy_node_info_dtype.
    :param move_array: The moves to be pushed, one for each of the structs in struct_array.  It is given as
    an ndarray with dtype np.uint8 and shape of [len(struct_array), 3] (dimention 2 has size 3 for
    from_square, to_square, and promotion).

    NOTES:
    1) While this function doesn't take up very much time, speed improvements should be considered a very
    high priority.  This is because unlike most other functions, the GPU will be idle (or at least very underutilized)
    during it's execution (This is due to it creating data for the GPU to consume)
        -I have a plan for a staged implementation in TensorFlow to avoid this entirely, but it would require the use
        of the C++ API, so it may take some time (but then it will also be able to be used from compiled functions).
         I plan to propose this idea somewhere in the GitHub repository (like in the wiki or issues sections) within
         the next few days.
    2) At this time I don't believe the big loop in this function is being vectorized by the LLVM compiler.  I think
    when some refactoring is done this may happen automatically (things like storing occupied_w and occupied_b
    as an array so it can be indexed with turn).
    """
    for j in range(len(struct_array)):
        move_from_square, move_to_square, move_promotion = _to_chess960_tuple(struct_array[j], move_array[j])

        # Reset ep square.
        ep_square = struct_array[j].ep_square
        struct_array[j].ep_square = 0

        # reset the ep square in the hash
        if ep_square != 0:
            if struct_array[j].turn == TURN_WHITE:
                ep_mask = shift_down(BB_SQUARES[ep_square])
                if (shift_left(ep_mask) | shift_right(ep_mask)) & struct_array[j].pawns & struct_array[j].occupied_w:
                    struct_array[j].hash ^= RANDOM_ARRAY[772 + square_file(ep_square)]
            else:
                ep_mask = shift_up(BB_SQUARES[ep_square])
                if (shift_left(ep_mask) | shift_right(ep_mask)) & struct_array[j].pawns & struct_array[j].occupied_b:
                    struct_array[j].hash ^= RANDOM_ARRAY[772 + square_file(ep_square)]

        # Increment move counters.
        struct_array[j].halfmove_clock += 1
        if struct_array[j].turn == TURN_BLACK:
            pivot = 0
        else:
            pivot = 1

        # Zero the half move clock.
        if scalar_is_zeroing(struct_array[j], move_from_square, move_to_square):
            struct_array[j].halfmove_clock = 0

        from_bb = BB_SQUARES[move_from_square]
        to_bb = BB_SQUARES[move_to_square]

        piece_type = _remove_piece_at(struct_array[j], move_from_square)

        # Remove the piece that's being moved from the hash
        struct_array[j].hash ^= RANDOM_ARRAY[((piece_type - 1) * 2 + pivot) * 64 + move_from_square]

        capture_square = move_to_square

        captured_piece_type = piece_type_at(struct_array[j], capture_square)

        castle_deltas = struct_array[j].castling_rights

        struct_array[j].castling_rights = struct_array[j].castling_rights & ~to_bb & ~from_bb

        castle_deltas ^= struct_array[j].castling_rights

        if castle_deltas:
            if castle_deltas & BB_A1:
                struct_array[j].hash ^= RANDOM_ARRAY[768 + 1]
            if castle_deltas & BB_H1:
                struct_array[j].hash ^= RANDOM_ARRAY[768]
            if castle_deltas & BB_A8:
                struct_array[j].hash ^= RANDOM_ARRAY[768 + 3]
            if castle_deltas & BB_H8:
                struct_array[j].hash ^= RANDOM_ARRAY[768 + 2]

        if piece_type == KING:
            castle_deltas = struct_array[j].castling_rights
            if struct_array[j].turn == TURN_WHITE:
                struct_array[j].castling_rights &= ~BB_RANK_1
                castle_deltas ^= struct_array[j].castling_rights
                if castle_deltas:
                    if castle_deltas & BB_A1:
                        struct_array[j].hash ^= RANDOM_ARRAY[768 + 1]
                    if castle_deltas & BB_H1:
                        struct_array[j].hash ^= RANDOM_ARRAY[768]
            else:
                struct_array[j].castling_rights &= ~BB_RANK_8
                castle_deltas ^= struct_array[j].castling_rights
                if castle_deltas:
                    if castle_deltas & BB_A8:
                        struct_array[j].hash ^= RANDOM_ARRAY[768 + 3]
                    if castle_deltas & BB_H8:
                        struct_array[j].hash ^= RANDOM_ARRAY[768 + 2]

        if piece_type == PAWN:
            if move_to_square >= move_from_square:
                diff = move_to_square - move_from_square
                if diff == 16:
                    struct_array[j].ep_square = move_from_square + 8
                elif ep_square != 0:
                    if move_to_square == ep_square and diff in [7, 9] and not captured_piece_type:
                        # Remove pawns captured en passant.
                        capture_square = ep_square - 8
                        remove_piece_mask = BB_SQUARES[capture_square]

                        struct_array[j].pawns ^= remove_piece_mask
                        struct_array[j].occupied ^= remove_piece_mask
                        struct_array[j].occupied_b &= ~remove_piece_mask

                        # Remove the captured pawn from the Zobrist hash
                        struct_array[j].hash ^= RANDOM_ARRAY[capture_square]
            else:
                diff = move_from_square - move_to_square
                if diff == 16:
                    struct_array[j].ep_square = move_from_square - 8
                elif ep_square != 0:
                    if move_to_square == ep_square and diff in [7, 9] and not captured_piece_type:
                        # Remove pawns captured en passant.
                        capture_square = ep_square + 8
                        remove_piece_mask = BB_SQUARES[capture_square]

                        struct_array[j].pawns ^= remove_piece_mask
                        struct_array[j].occupied ^= remove_piece_mask
                        struct_array[j].occupied_w &= ~remove_piece_mask

                        # Remove the captured pawn from the Zobrist hash
                        struct_array[j].hash ^= RANDOM_ARRAY[64 + capture_square]

        # Promotion.
        if move_promotion != 0:
            piece_type = move_promotion

        # Castling.
        if struct_array[j].turn:
            castling = piece_type == KING and struct_array[j].occupied_w & to_bb
        else:
            castling = piece_type == KING and struct_array[j].occupied_b & to_bb
        if castling:

            ###Should be directly implementing a special version of these since the types are known
            # (look up a few lines to pawn removal, something like that)####
            _remove_piece_at(struct_array[j], move_from_square)
            _remove_piece_at(struct_array[j], move_to_square)

            if square_file(move_to_square) < square_file(move_from_square):
                if struct_array[j].turn == TURN_WHITE:
                    _set_piece_at(struct_array[j], C1, KING, struct_array[j].turn)
                    _set_piece_at(struct_array[j], D1, ROOK, struct_array[j].turn)

                    struct_array[j].hash ^= RANDOM_ARRAY[((ROOK - 1) * 2 + pivot) * 64 + A1] ^ \
                                            RANDOM_ARRAY[((KING - 1) * 2 + pivot) * 64 + C1] ^ \
                                            RANDOM_ARRAY[((ROOK - 1) * 2 + pivot) * 64 + D1]
                else:
                    _set_piece_at(struct_array[j], C8, KING, struct_array[j].turn)
                    _set_piece_at(struct_array[j], D8, ROOK, struct_array[j].turn)

                    struct_array[j].hash ^= RANDOM_ARRAY[((ROOK - 1) * 2 + pivot) * 64 + A8] ^ \
                                            RANDOM_ARRAY[((KING - 1) * 2 + pivot) * 64 + C8] ^ \
                                            RANDOM_ARRAY[((ROOK - 1) * 2 + pivot) * 64 + D8]
            else:
                if struct_array[j].turn == TURN_WHITE:
                    _set_piece_at(struct_array[j], G1, KING, struct_array[j].turn)
                    _set_piece_at(struct_array[j], F1, ROOK, struct_array[j].turn)
                    struct_array[j].hash ^= RANDOM_ARRAY[((ROOK - 1) * 2 + pivot) * 64 + H1] ^ \
                                            RANDOM_ARRAY[((KING - 1) * 2 + pivot) * 64 + G1] ^ \
                                            RANDOM_ARRAY[((ROOK - 1) * 2 + pivot) * 64 + F1]
                else:
                    _set_piece_at(struct_array[j], G8, KING, struct_array[j].turn)
                    _set_piece_at(struct_array[j], F8, ROOK, struct_array[j].turn)

                    struct_array[j].hash ^= RANDOM_ARRAY[((ROOK - 1) * 2 + pivot) * 64 + H8] ^ \
                                            RANDOM_ARRAY[((KING - 1) * 2 + pivot) * 64 + G8] ^ \
                                            RANDOM_ARRAY[((ROOK - 1) * 2 + pivot) * 64 + F8]

        # Put piece on target square.
        if not castling and piece_type != 0:
            _set_piece_at(struct_array[j], move_to_square, piece_type, struct_array[j].turn)

            # Put the moving piece in the new location in the hash
            struct_array[j].hash ^= RANDOM_ARRAY[((piece_type - 1) * 2 + pivot) * 64 + move_to_square]

            if captured_piece_type:
                struct_array[j].hash ^= RANDOM_ARRAY[
                    ((captured_piece_type - 1) * 2 + (pivot + 1) % 2) * 64 + move_to_square]


        # Swap turn.
        struct_array[j].turn = not struct_array[j].turn
        struct_array[j].hash ^= RANDOM_ARRAY[780]

        # set the ep square in the hash
        if struct_array[j].ep_square != 0:
            if struct_array[j].turn == TURN_WHITE:
                ep_mask = shift_down(BB_SQUARES[struct_array[j].ep_square])
                if (shift_left(ep_mask) | shift_right(ep_mask)) & struct_array[j].pawns & struct_array[j].occupied_w:
                    struct_array[j].hash ^= RANDOM_ARRAY[772 + square_file(struct_array[j].ep_square)]
            else:
                ep_mask = shift_up(BB_SQUARES[struct_array[j].ep_square])
                if (shift_left(ep_mask) | shift_right(ep_mask)) & struct_array[j].pawns & struct_array[j].occupied_b:
                    struct_array[j].hash ^= RANDOM_ARRAY[772 + square_file(struct_array[j].ep_square)]





@njit
def _attackers_mask(board_state, color, square, occupied):
    queens_and_rooks = board_state.queens | board_state.rooks
    queens_and_bishops = board_state.queens | board_state.bishops

    if color == TURN_WHITE:
        color_index = BLACK
    else:
        color_index = WHITE
    attackers = (
        (BB_KING_ATTACKS[square] & board_state.kings) |
        (BB_KNIGHT_ATTACKS[square] & board_state.knights) |
        (RANK_ATTACK_ARRAY[square, khash_get(RANK_ATTACK_INDEX_LOOKUP_TABLE, BB_RANK_MASKS[square] & occupied, 0)] & queens_and_rooks) |
        (FILE_ATTACK_ARRAY[square, khash_get(FILE_ATTACK_INDEX_LOOKUP_TABLE, BB_FILE_MASKS[square] & occupied, 0)] & queens_and_rooks) |
        (DIAG_ATTACK_ARRAY[square, khash_get(DIAG_ATTACK_INDEX_LOOKUP_TABLE, BB_DIAG_MASKS[square] & occupied, 0)] & queens_and_bishops) |
        (BB_PAWN_ATTACKS[color_index, square] & board_state.pawns))

    if color == TURN_WHITE:
        return attackers & board_state.occupied_w
    return attackers & board_state.occupied_b


@njit
def attacks_mask(board_state, square):
    bb_square = BB_SQUARES[square]

    if bb_square & board_state.pawns:
        if bb_square & board_state.occupied_w:
            return BB_PAWN_ATTACKS[WHITE, square]
        else:
            return BB_PAWN_ATTACKS[BLACK, square]
    elif bb_square & board_state.knights:
        return BB_KNIGHT_ATTACKS[square]
    elif bb_square & board_state.kings:
        return BB_KING_ATTACKS[square]
    else:
        attacks = np.uint64(0)
        if bb_square & board_state.bishops or bb_square & board_state.queens:
            attacks = DIAG_ATTACK_ARRAY[square,
                khash_get(DIAG_ATTACK_INDEX_LOOKUP_TABLE, BB_DIAG_MASKS[square] & board_state.occupied, 0)]
        if bb_square & board_state.rooks or bb_square & board_state.queens:

            attacks |= (RANK_ATTACK_ARRAY[square,
                            khash_get(RANK_ATTACK_INDEX_LOOKUP_TABLE, BB_RANK_MASKS[square] & board_state.occupied,0)] |
                        FILE_ATTACK_ARRAY[square,
                            khash_get(FILE_ATTACK_INDEX_LOOKUP_TABLE, BB_FILE_MASKS[square] & board_state.occupied, 0)])
        return attacks



@njit
def _slider_blockers(board_state, king):
    snipers = (((board_state.rooks | board_state.queens) &
               (RANK_ATTACK_ARRAY[king, 0] | FILE_ATTACK_ARRAY[king, 0])) |
               (DIAG_ATTACK_ARRAY[king, 0] & (board_state.bishops | board_state.queens)))

    blockers = 0
    if board_state.turn == TURN_WHITE:
        for sniper in scan_reversed(snipers & board_state.occupied_b):
            b = BB_BETWEEN[king, sniper] & board_state.occupied

            # Add to blockers if exactly one piece in between.
            if b and BB_SQUARES[msb(b)] == b:
                blockers |= b

        return blockers & board_state.occupied_w
    else:
        for sniper in scan_reversed(snipers & board_state.occupied_w):
            b = BB_BETWEEN[king, sniper] & board_state.occupied

            # Add to blockers if exactly one piece in between.
            if b and BB_SQUARES[msb(b)] == b:
                blockers |= b

        return blockers & board_state.occupied_b


@njit
def _attacked_for_king(board_state, path, occupied):
    for sq in scan_reversed(path):
        if _attackers_mask(board_state, not board_state.turn, sq, occupied):
            return True
    return False


@njit
def _castling_uncovers_rank_attack(board_state, rook_bb, king_to):
    """
    Test the special case where we castle and our rook shielded us from
    an attack, so castling would be into check.
    """
    rank_pieces = BB_RANK_MASKS[king_to] & (board_state.occupied ^ rook_bb)
    if board_state.turn:
        sliders = (board_state.queens | board_state.rooks) & board_state.occupied_b
    else:
        sliders = (board_state.queens | board_state.rooks) & board_state.occupied_w
    return RANK_ATTACK_ARRAY[king_to, khash_get(RANK_ATTACK_INDEX_LOOKUP_TABLE, rank_pieces, 0)] & sliders


@njit
def _from_chess960_tuple(board_state, from_square, to_square):
    if from_square == E1 and board_state.kings & BB_E1:
        if to_square == H1:
            return E1, G1, np.uint8(0)
        elif to_square == A1:
            return E1, C1, np.uint8(0)
    elif from_square == E8 and board_state.kings & BB_E8:
        if to_square == H8:
            return E8, G8, np.uint8(0)
        elif to_square == A8:
            return E8, C8, np.uint8(0)

    # promotion is set to 0 because this function only gets called when looking for castling moves,
    # which can't have promotions
    return from_square, to_square, np.uint8(0)




@njit
def set_pseudo_legal_ep(board_state, from_mask=BB_ALL, to_mask=BB_ALL):
    if board_state['ep_square'] == 0:
        return

    if not BB_SQUARES[board_state['ep_square']] & to_mask:
        return

    if BB_SQUARES[board_state['ep_square']] & board_state['occupied']:
        return

    if board_state['turn']:
        capturers = (
            board_state['pawns'] & board_state['occupied_w'] & from_mask &
            BB_PAWN_ATTACKS[BLACK, board_state['ep_square']] & BB_RANKS[4])
    else:
        capturers = (
            board_state['pawns'] & board_state['occupied_b'] & from_mask &
            BB_PAWN_ATTACKS[WHITE, board_state['ep_square']] & BB_RANKS[3])

    for capturer in scan_reversed(capturers):
        board_state['unexplored_moves'][board_state.children_left] = np.array([capturer, board_state['ep_square'], 0],dtype=np.uint8)
        board_state['children_left'] += 1


@njit
def set_castling_moves(board_state, from_mask=BB_ALL, to_mask=BB_ALL):
    if board_state['turn']:
        backrank = BB_RANK_1
        king = board_state['occupied_w'] & board_state['kings'] & backrank & from_mask
    else:
        backrank = BB_RANK_8
        king = board_state['occupied_b'] & board_state['kings'] & backrank & from_mask

    king = king & -king

    if not king or _attacked_for_king(board_state, king, board_state['occupied']):
        return

    for candidate in scan_reversed(board_state['castling_rights'] & backrank & to_mask):
        rook = BB_SQUARES[candidate]

        empty_for_rook = np.uint64(0)
        empty_for_king = np.uint64(0)

        if rook < king:
            bb_c = BB_FILE_C & backrank
            bb_d = BB_FILE_D & backrank

            king_to = msb(bb_c)
            if not rook & bb_d:
                empty_for_rook = BB_BETWEEN[candidate, msb(bb_d)] | bb_d
            if not king & bb_c:
                empty_for_king = BB_BETWEEN[msb(king), king_to] | bb_c
        else:
            bb_f = BB_FILE_F & backrank
            bb_g = BB_FILE_G & backrank

            king_to = msb(bb_g)
            if not rook & bb_f:
                empty_for_rook = BB_BETWEEN[candidate, msb(bb_f)] | bb_f
            if not king & bb_g:
                empty_for_king = BB_BETWEEN[msb(king), king_to] | bb_g

        if not ((board_state['occupied'] ^ king ^ rook) & (empty_for_king | empty_for_rook) or
                _attacked_for_king(board_state, empty_for_king, board_state['occupied'] ^ king) or
                _castling_uncovers_rank_attack(board_state, rook, king_to)):
            board_state['unexplored_moves'][board_state.children_left, :] = _from_chess960_tuple(board_state, msb(king), candidate)
            board_state['children_left'] += 1


@njit
def set_pseudo_legal_moves(board_state, from_mask=BB_ALL, to_mask=BB_ALL):
    """
    NOTES:
    1) All of the NumPy array creation when setting the unexplored_moves may or may not have speed penalties,
    but either way it would be much better to somehow just convince Numba that the values have
    dtype np.uint8 (which they do!).
    """
    if board_state['turn'] == TURN_WHITE:
        cur_turn_occupied = board_state['occupied_w']
        opponent_occupied = board_state['occupied_b']
    else:
        cur_turn_occupied = board_state['occupied_b']
        opponent_occupied = board_state['occupied_w']

    our_pieces = cur_turn_occupied

    # Generate piece moves.
    non_pawns = our_pieces & ~board_state['pawns'] & from_mask

    for from_square in scan_reversed(non_pawns):

        moves = attacks_mask(board_state, from_square) & ~our_pieces & to_mask
        for to_square in scan_reversed(moves):
            board_state['unexplored_moves'][board_state.children_left, :] = np.array([from_square, to_square, 0],dtype=np.uint8)
            board_state['children_left'] += 1

    # Generate castling moves.
    if from_mask & board_state['kings']:
        set_castling_moves(board_state, from_mask, to_mask)

    # The remaining moves are all pawn moves.
    pawns = board_state['pawns'] & cur_turn_occupied & from_mask
    if not pawns:
        return

    # Generate pawn captures.
    capturers = pawns
    for from_square in scan_reversed(capturers):
        if board_state['turn']:
            targets = BB_PAWN_ATTACKS[WHITE, from_square] & opponent_occupied & to_mask
        else:
            targets = BB_PAWN_ATTACKS[BLACK, from_square] & opponent_occupied & to_mask

        for to_square in scan_reversed(targets):
            if square_rank(to_square) in [0, 7]:
                board_state['unexplored_moves'][board_state.children_left:board_state.children_left+4, 0] = from_square
                board_state['unexplored_moves'][board_state.children_left:board_state.children_left+4, 1] = to_square
                board_state['unexplored_moves'][board_state.children_left:board_state.children_left+4, 2] = (QUEEN, ROOK, BISHOP, KNIGHT)
                board_state['children_left'] += 4
            else:
                board_state['unexplored_moves'][board_state.children_left, :] = np.array([from_square, to_square, 0],dtype=np.uint8)
                board_state['children_left'] += 1

    # Prepare pawn advance generation.
    if board_state['turn'] == TURN_WHITE:
        single_moves = pawns << 8 & ~board_state['occupied']
        double_moves = single_moves << 8 & ~board_state['occupied'] & (BB_RANK_3 | BB_RANK_4)
    else:
        single_moves = pawns >> 8 & ~board_state['occupied']
        double_moves = single_moves >> 8 & ~board_state['occupied'] & (BB_RANK_6 | BB_RANK_5)

    single_moves &= to_mask
    double_moves &= to_mask

    # Generate single pawn moves.
    for to_square in scan_reversed(single_moves):
        if board_state['turn'] == TURN_BLACK:
            from_square = to_square + 8
        else:
            from_square = to_square - 8

        if square_rank(to_square) in [0, 7]:
            board_state['unexplored_moves'][board_state.children_left:board_state.children_left + 4, 0] = from_square
            board_state['unexplored_moves'][board_state.children_left:board_state.children_left + 4, 1] = to_square
            board_state['unexplored_moves'][board_state.children_left:board_state.children_left + 4, 2] = (QUEEN, ROOK, BISHOP, KNIGHT)
            board_state['children_left'] += 4
        else:
            board_state['unexplored_moves'][board_state.children_left, :] = np.array([from_square, to_square, 0],dtype=np.uint8)
            board_state['children_left'] += 1

    # Generate double pawn moves.
    for to_square in scan_reversed(double_moves):
        if board_state['turn'] == TURN_BLACK:
            from_square = to_square + 16
        else:
            from_square = to_square - 16

        board_state['unexplored_moves'][board_state.children_left,:] = np.array([from_square, to_square, 0],dtype=np.uint8)
        board_state['children_left'] += 1

    # Generate en passant captures.
    if board_state['ep_square'] != 0:
        set_pseudo_legal_ep(board_state, from_mask, to_mask)


@njit
def pin_mask(board_state, color, square):
    if color:
        king = msb(board_state.occupied_w & board_state.kings)
    else:
        king = msb(board_state.occupied_b & board_state.kings)

    square_mask = BB_SQUARES[square]
    for attacks, sliders in zip((FILE_ATTACK_ARRAY, RANK_ATTACK_ARRAY, DIAG_ATTACK_ARRAY), (
        board_state.rooks | board_state.queens, board_state.rooks | board_state.queens,
        board_state.bishops | board_state.queens)):
        rays = attacks[king][0]
        if rays & square_mask:
            if color:
                snipers = rays & sliders & board_state.occupied_b
            else:
                snipers = rays & sliders & board_state.occupied_w

            for sniper in scan_reversed(snipers):
                if BB_BETWEEN[sniper, king] & (board_state.occupied | square_mask) == square_mask:
                    return BB_RAYS[king, sniper]

            break

    return BB_ALL




@njit
def is_pseudo_legal_ep(board_state, from_mask, to_mask):
    if board_state['ep_square'] == 0:
        return False

    if not BB_SQUARES[board_state['ep_square']] & to_mask:
        return False

    if BB_SQUARES[board_state['ep_square']] & board_state['occupied']:
        return False

    if board_state['turn']:
        if board_state['pawns'] & board_state['occupied_w'] & from_mask & BB_PAWN_ATTACKS[BLACK, board_state['ep_square']] & BB_RANKS[4] != 0:
            return True
    else:
        if board_state['pawns'] & board_state['occupied_b'] & from_mask & BB_PAWN_ATTACKS[WHITE, board_state['ep_square']] & BB_RANKS[3] != 0:
            return True

    return False



@njit
def is_pseudo_legal_castling_move(board_state, from_mask, to_mask):
    if board_state['turn']:
        backrank = BB_RANK_1
        king = board_state['occupied_w'] & board_state['kings'] & backrank & from_mask
    else:
        backrank = BB_RANK_8
        king = board_state['occupied_b'] & board_state['kings'] & backrank & from_mask

    king = king & -king

    if not king or _attacked_for_king(board_state, king, board_state['occupied']):
        return False


    candidates = board_state['castling_rights'] & backrank & to_mask
    if candidates != 0:
        candidate = msb(candidates)
        rook = BB_SQUARES[candidate]

        empty_for_rook = np.uint64(0)
        empty_for_king = np.uint64(0)

        if rook < king:
            bb_c = BB_FILE_C & backrank
            bb_d = BB_FILE_D & backrank


            king_to = msb(bb_c)
            if not rook & bb_d:
                empty_for_rook = BB_BETWEEN[candidate, msb(bb_d)] | bb_d
            if not king & bb_c:
                empty_for_king = BB_BETWEEN[msb(king), king_to] | bb_c
        else:
            bb_f = BB_FILE_F & backrank
            bb_g = BB_FILE_G & backrank


            king_to = msb(bb_g)
            if not rook & bb_f:
                empty_for_rook = BB_BETWEEN[candidate, msb(bb_f)] | bb_f
            if not king & bb_g:
                empty_for_king = BB_BETWEEN[msb(king), king_to] | bb_g

        if not ((board_state['occupied'] ^ king ^ rook) & (empty_for_king | empty_for_rook) or
                _attacked_for_king(board_state, empty_for_king, board_state['occupied'] ^ king) or
                _castling_uncovers_rank_attack(board_state, rook, king_to)):
            return True

    return False


@njit
def scalar_is_pseudo_legal_move(board_state, move):
    if board_state['turn'] == TURN_WHITE:
        cur_turn_occupied = board_state['occupied_w']
        opponent_occupied = board_state['occupied_b']
    else:
        cur_turn_occupied = board_state['occupied_b']
        opponent_occupied = board_state['occupied_w']

    our_pieces = cur_turn_occupied

    from_mask = BB_SQUARES[move[0]]
    to_mask = BB_SQUARES[move[1]]

    # Generate piece moves.
    non_pawns = our_pieces & ~board_state['pawns'] & from_mask
    if non_pawns != 0:
        if attacks_mask(board_state, msb(non_pawns)) & ~our_pieces & to_mask != 0:
            if move[2] == 0:
                return True

    # Generate castling moves.
    if from_mask & board_state['kings']:

        chess960_from_square, chess960_to_square, _ = _to_chess960_tuple(board_state, move)
        if chess960_to_square != move[1]:
            if is_pseudo_legal_castling_move(board_state, BB_SQUARES[chess960_from_square], BB_SQUARES[chess960_to_square]):
                if move[2] == 0:
                    return True

    # The remaining possible moves are all pawn moves.
    pawns = board_state['pawns'] & cur_turn_occupied & from_mask
    if not pawns:
        return False


    # Check pawn captures.
    if board_state['turn']:
        targets = BB_PAWN_ATTACKS[WHITE, msb(pawns)] & opponent_occupied & to_mask
        if targets != 0:
            if square_rank(msb(targets)) in [0,7]:
                if not move[2] in [0, PAWN, KING]:
                    return True
                else:
                    return False
            else:
                if move[2] == 0:
                    return True
                else:
                    return False
    else:
        if BB_PAWN_ATTACKS[BLACK, msb(pawns)] & opponent_occupied & to_mask != 0:
            return True



    # Check pawn advance generation.
    if board_state['turn']:
        single_moves = pawns << 8 & ~board_state['occupied']
        if single_moves & to_mask:

            if square_rank(move[1]) in [0,7]:
                if not move[2] in [0, PAWN, KING]:
                    return True
            else:
                if move[2] == 0:
                    return True
        if (single_moves << 8 & ~board_state['occupied'] & (BB_RANK_3 | BB_RANK_4)) & to_mask:
            if move[2] == 0:
                return True
    else:
        single_moves = pawns >> 8 & ~board_state['occupied']
        if single_moves & to_mask:

            if square_rank(move[1]) in [0,7]:
                if not move[2] in [0, PAWN, KING]:
                    return True
            else:
                if move[2] == 0:
                    return True
        if (single_moves >> 8 & ~board_state['occupied'] & (BB_RANK_6 | BB_RANK_5)) & to_mask:
            if move[2] == 0:
                return True

    # Generate en passant captures.
    if board_state['ep_square'] != 0:
        if move[2] == 0:
            return is_pseudo_legal_ep(board_state, from_mask, to_mask)
    return False






@njit
def is_evasion(board_state, king, checkers, from_mask, to_mask):
    """
    NOTES:
    1) This does NOT check if the move is legal
    """
    sliders = checkers & (board_state.bishops | board_state.rooks | board_state.queens)

    attacked = np.uint64(0)
    for checker in scan_reversed(sliders):
        attacked |= BB_RAYS[king, checker] & ~BB_SQUARES[checker]

    if BB_SQUARES[king] & from_mask:
        if board_state.turn:
            if BB_KING_ATTACKS[king] & ~board_state.occupied_w & ~attacked & to_mask != 0:
                return True

        else:
            if BB_KING_ATTACKS[king] & ~board_state.occupied_b & ~attacked & to_mask != 0:
                return True

    checker = msb(checkers)
    if BB_SQUARES[checker] == checkers:
        # If it captures or blocks a single checker.
        return ~board_state.kings & from_mask and (BB_BETWEEN[king, checker] | checkers) & to_mask

    return False



@njit
def set_evasions(board_state, king, checkers, from_mask=BB_ALL, to_mask=BB_ALL):
    sliders = checkers & (board_state.bishops | board_state.rooks | board_state.queens)

    attacked = np.uint64(0)
    for checker in scan_reversed(sliders):
        attacked |= BB_RAYS[king, checker] & ~BB_SQUARES[checker]

    if BB_SQUARES[king] & from_mask:
        if board_state.turn:
            for to_square in scan_reversed(BB_KING_ATTACKS[king] & ~board_state.occupied_w & ~attacked & to_mask):
                board_state['unexplored_moves'][board_state.children_left, 0] = king
                board_state['unexplored_moves'][board_state.children_left, 1] = to_square
                board_state['unexplored_moves'][board_state.children_left, 2] = 0
                board_state['children_left'] += 1
        else:
            for to_square in scan_reversed(BB_KING_ATTACKS[king] & ~board_state.occupied_b & ~attacked & to_mask):
                board_state['unexplored_moves'][board_state.children_left, 0] = king
                board_state['unexplored_moves'][board_state.children_left, 1] = to_square
                board_state['unexplored_moves'][board_state.children_left, 2] = 0
                board_state['children_left'] += 1

    checker = msb(checkers)
    if BB_SQUARES[checker] == checkers:
        # Capture or block a single checker.
        target = BB_BETWEEN[king][checker] | checkers

        set_pseudo_legal_moves(board_state, ~board_state.kings & from_mask, target & to_mask)

        # Capture the checking pawn en passant (but avoid yielding duplicate moves).
        if board_state.ep_square != 0:
            if not BB_SQUARES[board_state.ep_square] & target:
                if board_state.turn == TURN_BLACK:
                    last_double = board_state.ep_square + 8
                else:
                    last_double = board_state.ep_square - 8
                if last_double == checker:
                    set_pseudo_legal_ep(board_state, from_mask, to_mask)


@njit
def new_is_castling(board_state, from_square, to_square):
    """
    Checks if the given pseudo-legal move is a castling move.
    """
    if board_state.kings & BB_SQUARES[from_square]:
        from_file = square_file(from_square)
        to_file = square_file(to_square)
        if from_file > to_file:
            diff = from_file - to_file
        else:
            diff = to_file - from_file

        if board_state.turn == TURN_WHITE:
            return diff > 1 or bool(board_state.rooks & board_state.occupied_w & BB_SQUARES[to_square])
        else:
            return diff > 1 or bool(board_state.rooks & board_state.occupied_b & BB_SQUARES[to_square])

    return False





@njit
def new_is_en_passant(board_state, from_square, to_square):
    """
    Checks if the given pseudo-legal move is an en passant capture.
    """
    if board_state.ep_square != 0:
        if to_square >= from_square:
            return (board_state.ep_square == to_square and
                    bool(board_state.pawns & BB_SQUARES[from_square]) and
                    to_square - from_square in [7, 9] and
                    not board_state.occupied & BB_SQUARES[to_square])
        else:
            return (board_state.ep_square == to_square and
                    bool(board_state.pawns & BB_SQUARES[from_square]) and
                    from_square - to_square in [7, 9] and
                    not board_state.occupied & BB_SQUARES[to_square])

    return False



@njit
def _new_ep_skewered(board_state, king, capturer):
    """
    Handle the special case where the king would be in check, if the pawn and its capturer disappear from the rank.

    Vertical skewers of the captured pawn are not possible. (Pins on the capturer are not handled here.)
    """
    # only using as workaround, won't do this long term
    if board_state.turn:
        last_double = board_state.ep_square - 8
        occupancy = (board_state.occupied & ~BB_SQUARES[last_double] & ~BB_SQUARES[capturer] | BB_SQUARES[board_state.ep_square])

        # Horizontal attack on the fifth or fourth rank.
        horizontal_attackers = board_state.occupied_b & (board_state.rooks | board_state.queens)
        if RANK_ATTACK_ARRAY[king, khash_get(RANK_ATTACK_INDEX_LOOKUP_TABLE, BB_RANK_MASKS[king] & occupancy, 0)] & horizontal_attackers:
            return True

    else:
        last_double = board_state.ep_square + 8
        occupancy = (board_state.occupied & ~BB_SQUARES[last_double] & ~BB_SQUARES[capturer] | BB_SQUARES[board_state.ep_square])

        # Horizontal attack on the fifth or fourth rank.
        horizontal_attackers = board_state.occupied_w & (board_state.rooks | board_state.queens)
        if RANK_ATTACK_ARRAY[king,
                             khash_get(
                                 RANK_ATTACK_INDEX_LOOKUP_TABLE,
                                 BB_RANK_MASKS[king] & occupancy,
                                 0)] & horizontal_attackers:
            return True

    return False



@njit
def new_is_safe(board_state, king, blockers, from_square, to_square):
    if from_square == king:
        if new_is_castling(board_state, from_square, to_square):
            return True
        else:
            return not bool(_attackers_mask(board_state, not board_state.turn, to_square, board_state.occupied))

    elif new_is_en_passant(board_state, from_square, to_square):
        return (pin_mask(board_state, board_state.turn, from_square) & BB_SQUARES[to_square] and
                not _new_ep_skewered(board_state, king, from_square))
    else:
        return (not blockers & BB_SQUARES[from_square] or
                BB_RAYS[from_square, to_square] & BB_SQUARES[king])


@njit
def set_up_move_array(board_struct):
    if board_struct.turn == TURN_WHITE:
        king = msb(board_struct['kings'] & board_struct['occupied_w'])
    else:
        king = msb(board_struct['kings'] & board_struct['occupied_b'])

    blockers = _slider_blockers(board_struct, king)
    checkers = _attackers_mask(board_struct, not board_struct['turn'], king, board_struct['occupied'])


    # If in check
    if checkers:
        set_evasions(board_struct, king, checkers, BB_ALL, BB_ALL)

        if board_struct['children_left']==0:
            board_struct['terminated'] = True
            board_struct['best_value'] = LOSS_RESULT_SCORE
            #LOOK INTO IF IT CAN RETURN HERE (not set illigal moves to 255)

    else:
        set_pseudo_legal_moves(board_struct, BB_ALL, BB_ALL)

        if board_struct['children_left']==0:
            board_struct['terminated'] = True
            board_struct['best_value'] = TIE_RESULT_SCORE
            # LOOK INTO IF IT CAN RETURN HERE (not set illigal moves to 255)



    legal_move_index = 0
    for j in range(board_struct['children_left']):
        if new_is_safe(board_struct, king, blockers, board_struct['unexplored_moves'][j, 0], board_struct['unexplored_moves'][j, 1]):
            board_struct['unexplored_moves'][legal_move_index] = board_struct['unexplored_moves'][j]
            legal_move_index += 1

    board_struct['unexplored_moves'][legal_move_index:board_struct['children_left'],:] = 255
    board_struct['children_left'] = legal_move_index


@njit
def set_up_move_array_except_move(board_struct, move_to_avoid):
    if board_struct['turn'] == TURN_WHITE:
        king = msb(board_struct['kings'] & board_struct['occupied_w'])
    else:
        king = msb(board_struct['kings'] & board_struct['occupied_b'])

    blockers = _slider_blockers(board_struct, king)
    checkers = _attackers_mask(board_struct, not board_struct['turn'], king, board_struct['occupied'])

    # If in check
    if checkers:
        set_evasions(board_struct, king, checkers, BB_ALL, BB_ALL)
    else:
        set_pseudo_legal_moves(board_struct, BB_ALL, BB_ALL)

    legal_move_index = 0
    for j in range(board_struct['children_left']):
        if np.any(move_to_avoid != board_struct['unexplored_moves'][j]):
            if new_is_safe(board_struct, king, blockers, board_struct['unexplored_moves'][j,0],board_struct['unexplored_moves'][j,1]):
                board_struct['unexplored_moves'][legal_move_index] = board_struct['unexplored_moves'][j]
                legal_move_index += 1

    board_struct['unexplored_moves'][legal_move_index:board_struct['children_left'],:] = 255
    board_struct['children_left'] = legal_move_index



@njit
def scalar_is_into_check(board_scalar, from_square, to_square):
    """
    Checks if the given move would leave the king in check or put it into
    check. The move must be at least pseudo legal.  This function was adapted from the Python-Chess version of it.
    """
    if board_scalar.turn == TURN_WHITE:
        king = msb(board_scalar.occupied_w & board_scalar.kings)
    else:
        king = msb(board_scalar.occupied_b & board_scalar.kings)

    checkers = _attackers_mask(board_scalar, not board_scalar.turn, king, board_scalar.occupied)
    if checkers:
        return not is_evasion(board_scalar, king, checkers, BB_SQUARES[from_square], BB_SQUARES[to_square])

    return not new_is_safe(board_scalar, king, _slider_blockers(board_scalar, king), from_square, to_square)


@njit
def scalar_is_legal_move(board_scalar, move):
    return scalar_is_pseudo_legal_move(board_scalar, move) and not scalar_is_into_check(board_scalar, move[0], move[1])


@njit
def structured_scalar_perft_test_move_gen_helper(struct_array):
    for j in range(len(struct_array)):
        if struct_array[j]['turn'] == TURN_WHITE:
            king = msb(struct_array[j]['kings'] & struct_array[j]['occupied_w'])
        else:
            king = msb(struct_array[j]['kings'] & struct_array[j]['occupied_b'])

        blockers = _slider_blockers(struct_array[j], king)
        checkers = _attackers_mask(struct_array[j], not struct_array[j]['turn'], king, struct_array[j]['occupied'])

        if checkers:
            set_evasions(struct_array[j], king, checkers, BB_ALL, BB_ALL)
        else:
            set_pseudo_legal_moves(struct_array[j], BB_ALL, BB_ALL)

        legal_move_index = 0
        for i in range(struct_array[j]['children_left']):
            if new_is_safe(struct_array[j], king, blockers, struct_array[j]['unexplored_moves'][i, 0], struct_array[j]['unexplored_moves'][i, 1]):
                struct_array[j]['unexplored_moves'][legal_move_index] = struct_array[j]['unexplored_moves'][i]
                legal_move_index += 1

        struct_array[j]['unexplored_moves'][legal_move_index:struct_array[j]['children_left'], :] = 255
        struct_array[j]['children_left'] = legal_move_index




# @njit
def structured_scalar_perft_test(struct_array, depth, print_info=False):
    if print_info:
        print("Starting depth %d with %d nodes."%(depth, len(struct_array)))
    if depth==0:
        return len(struct_array)

    struct_array['unexplored_moves'] = np.full_like(struct_array['unexplored_moves'],255)
    struct_array['children_left'] = np.zeros_like(struct_array['children_left'])

    structured_scalar_perft_test_move_gen_helper(struct_array)


    if depth == 1:
        return np.sum(struct_array['children_left'])

    legal_moves = struct_array['unexplored_moves'][struct_array['unexplored_moves'][..., 0] != 255]


    repeated_struct_array = np.repeat(struct_array, struct_array['children_left'])

    #Not sure if these actually provide a speed increase
    repeated_struct_array = np.ascontiguousarray(repeated_struct_array)
    legal_moves = np.ascontiguousarray(legal_moves)

    push_moves(repeated_struct_array, legal_moves)

    return structured_scalar_perft_test(repeated_struct_array, depth-1)


