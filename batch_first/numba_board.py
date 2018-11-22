from .classes_and_structs import *



def convert_board_to_whites_perspective(ary):
    """
    NOTES:
    1) This doesn't change any moves which are stored
    """
    if ary[0]['turn']:
        return ary

    struct = ary[0]

    struct['occupied_co'] = flip_vertically(struct['occupied_co'][::-1])
    struct['occupied'] = flip_vertically(struct['occupied'])

    struct['kings'] = flip_vertically(struct['kings'])
    struct['queens'] = flip_vertically(struct['queens'])
    struct['rooks'] = flip_vertically(struct['rooks'])
    struct['bishops'] = flip_vertically(struct['bishops'])
    struct['knights'] = flip_vertically(struct['knights'])
    struct['pawns'] = flip_vertically(struct['pawns'])

    struct['castling_rights'] = flip_vertically(struct['castling_rights'])
    struct['ep_square'] = square_mirror(struct['ep_square']) if struct['ep_square'] != NO_EP_SQUARE else NO_EP_SQUARE
    struct['turn'] = True

    return ary


# Obviously needs to be refactored
@nb.vectorize([nb.uint8(nb.uint64)], nopython=True)
def msb(n):
    r = 0
    n = n >> 1
    while n:
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

@nb.vectorize([nb.uint8(nb.uint64)], nopython=True)
def popcount(n):
    n = n - ((n >> 1) & popcount_const_1)
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

NOT_BB_FILE_A = ~BB_FILE_A
NOT_BB_FILE_H = ~BB_FILE_H

@njit(nb.uint64(nb.uint64))
def shift_right(b):
    return (b << 1) & NOT_BB_FILE_A

@njit(nb.uint64(nb.uint64))
def shift_left(b):
    return (b >> 1) & NOT_BB_FILE_H

@nb.vectorize([nb.uint8(nb.uint8)], nopython=True)
def square_mirror(square):
    """Mirrors the square vertically."""
    return square ^ 0x38


@njit
def any(iterable):
    for _ in iterable:
        return True
    return False


CASTLING_DIFF_SQUARES = np.array([
    [[H8, G8, F8],
     [H1, G1, F1]],
    [[A8, C8, D8],
     [A1, C1, D1]]], dtype=np.uint64)


CASTLING_ZORBRIST_HASH_CHANGES = np.zeros([2, 2, 2], dtype=np.uint64)
for j in range(2):
    for color in COLORS:
        for pivot in range(2):
            CASTLING_ZORBRIST_HASH_CHANGES[j, color, pivot] = np.bitwise_xor.reduce(
                RANDOM_ARRAY[((np.array([ROOK, KING, ROOK], dtype=np.uint64) - 1) * 2 + pivot) * 64 + CASTLING_DIFF_SQUARES[j, color]])



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

    board_state.occupied_co[:] &= ~mask

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

    board_state.occupied_co[color] ^= mask


@njit
def has_insufficient_material(board_state):
    # Enough material to mate.
    if board_state['pawns'] or board_state['rooks'] or board_state['queens']:
        return False

    # A single knight or a single bishop.
    elif popcount(board_state['occupied']) <= 3:
        return True

    # More than a single knight.
    elif board_state['knights']:
        return False

    # All bishops on the same color.
    elif board_state['bishops'] & BB_DARK_SQUARES == 0:
        return True
    elif board_state['bishops'] & BB_LIGHT_SQUARES == 0:
        return True

    return False


@njit
def is_zeroing(board_state, move_from_square, move_to_square):
    """
    Checks if the given pseudo-legal move is a capture or pawn move.
    """
    return np.bool_(BB_SQUARES[move_from_square] & board_state.pawns or BB_SQUARES[move_to_square] & board_state.occupied_co[1 ^ board_state.turn])


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
        if ep_square:
            if struct_array[j].turn:
                ep_mask = shift_down(BB_SQUARES[ep_square])
            else:
                ep_mask = shift_up(BB_SQUARES[ep_square])

            if (shift_left(ep_mask) | shift_right(ep_mask)) & struct_array[j].pawns & struct_array[j].occupied_co[struct_array[j].turn]:
                struct_array[j].hash ^= RANDOM_ARRAY[772 + square_file(ep_square)]

        # Increment move counters.
        struct_array[j].halfmove_clock += 1

        pivot = 1 if struct_array[j].turn else 0

        # Zero the half move clock.
        if is_zeroing(struct_array[j], move_from_square, move_to_square):
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
            if struct_array[j].turn:
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
                elif ep_square:
                    if move_to_square == ep_square and diff in [7, 9] and not captured_piece_type:
                        # Remove pawns captured en passant.
                        capture_square = ep_square - 8
                        remove_piece_mask = BB_SQUARES[capture_square]

                        struct_array[j].pawns ^= remove_piece_mask
                        struct_array[j].occupied ^= remove_piece_mask
                        struct_array[j].occupied_co[BLACK] &= ~remove_piece_mask

                        # Remove the captured pawn from the Zobrist hash
                        struct_array[j].hash ^= RANDOM_ARRAY[capture_square]
            else:
                diff = move_from_square - move_to_square
                if diff == 16:
                    struct_array[j].ep_square = move_from_square - 8
                elif ep_square:
                    if move_to_square == ep_square and diff in [7, 9] and not captured_piece_type:
                        # Remove pawns captured en passant.
                        capture_square = ep_square + 8
                        remove_piece_mask = BB_SQUARES[capture_square]

                        struct_array[j].pawns ^= remove_piece_mask
                        struct_array[j].occupied ^= remove_piece_mask
                        struct_array[j].occupied_co[WHITE] &= ~remove_piece_mask

                        # Remove the captured pawn from the Zobrist hash
                        struct_array[j].hash ^= RANDOM_ARRAY[64 + capture_square]

        # Promotion.
        if move_promotion:
            piece_type = move_promotion

        # Castling.
        castling = piece_type == KING and struct_array[j].occupied_co[struct_array[j].turn] & to_bb

        if castling:
            # This could be using a special implementation since the types of pieces are known
            # (look up a few lines to pawn removal, for reference)
            _remove_piece_at(struct_array[j], move_from_square)
            _remove_piece_at(struct_array[j], move_to_square)

            temp_index1 = np.int8(square_file(move_to_square) < square_file(move_from_square))
            temp_index2 = struct_array[j].turn
            for the_square, the_piece in zip(CASTLING_DIFF_SQUARES[temp_index1, temp_index2, 1:], [KING, ROOK]):
                _set_piece_at(struct_array[j], the_square, the_piece, struct_array[j].turn)

            struct_array[j].hash ^= CASTLING_ZORBRIST_HASH_CHANGES[temp_index1, temp_index2, pivot]


        # Put piece on target square.
        if not castling and piece_type:
            _set_piece_at(struct_array[j], move_to_square, piece_type, struct_array[j].turn)

            # Put the moving piece in the new location in the hash
            struct_array[j].hash ^= RANDOM_ARRAY[((piece_type - 1) * 2 + pivot) * 64 + move_to_square]

            if captured_piece_type:
                struct_array[j].hash ^= RANDOM_ARRAY[
                    ((captured_piece_type - 1) * 2 + (pivot + 1) % 2) * 64 + move_to_square]


        # Swap turn.
        struct_array[j].turn ^= 1
        struct_array[j].hash ^= RANDOM_ARRAY[780]

        # set the ep square in the hash
        if struct_array[j].ep_square:
            if struct_array[j].turn:
                ep_mask = shift_down(BB_SQUARES[struct_array[j].ep_square])
            else:
                ep_mask = shift_up(BB_SQUARES[struct_array[j].ep_square])
            if (shift_left(ep_mask) | shift_right(ep_mask)) & struct_array[j].pawns & struct_array[j].occupied_co[struct_array[j].turn]:
                struct_array[j].hash ^= RANDOM_ARRAY[772 + square_file(struct_array[j].ep_square)]


@njit
def _attackers_mask(board_state, color, square, occupied):
    queens_and_rooks = board_state.queens | board_state.rooks
    queens_and_bishops = board_state.queens | board_state.bishops

    attackers = (
        (BB_KING_ATTACKS[square] & board_state.kings) |
        (BB_KNIGHT_ATTACKS[square] & board_state.knights) |
        (RANK_ATTACK_ARRAY[
             square,
             khash_get(RANK_ATTACK_INDEX_LOOKUP_TABLE, BB_RANK_MASKS[square] & occupied, 0)] & queens_and_rooks) |
        (FILE_ATTACK_ARRAY[
             square,
             khash_get(FILE_ATTACK_INDEX_LOOKUP_TABLE, BB_FILE_MASKS[square] & occupied, 0)] & queens_and_rooks) |
        (DIAG_ATTACK_ARRAY[
             square,
             khash_get(DIAG_ATTACK_INDEX_LOOKUP_TABLE, BB_DIAG_MASKS[square] & occupied, 0)] & queens_and_bishops) |
        (BB_PAWN_ATTACKS[1 ^ color, square] & board_state.pawns))

    return attackers & board_state.occupied_co[color]


@njit
def attacks_mask(board_state, square):
    bb_square = BB_SQUARES[square]

    if bb_square & board_state.pawns:
        return BB_PAWN_ATTACKS[np.int8(bb_square & board_state.occupied_co[WHITE]), square]
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
def pin_mask(board_state, color, square):
    king = msb(board_state.occupied_co[color] & board_state.kings)
    square_mask = BB_SQUARES[square]
    for attacks, sliders in zip((FILE_ATTACK_ARRAY, RANK_ATTACK_ARRAY, DIAG_ATTACK_ARRAY), (
        board_state.rooks | board_state.queens, board_state.rooks | board_state.queens,
        board_state.bishops | board_state.queens)):
        rays = attacks[king][0]
        if rays & square_mask:
            snipers = rays & sliders & board_state.occupied_co[1 ^ color]

            for sniper in scan_reversed(snipers):
                if BB_BETWEEN[sniper, king] & (board_state.occupied | square_mask) == square_mask:
                    return BB_RAYS[king, sniper]
            break

    return BB_ALL


@njit
def is_en_passant(board_state, from_square, to_square):
    """
    Checks if the given pseudo-legal move is an en passant capture.
    """
    return board_state.ep_square and (
            board_state.ep_square == to_square and board_state.pawns & BB_SQUARES[from_square] and np.abs(np.int8(
        to_square - from_square)) in [7, 9] and not board_state.occupied & BB_SQUARES[to_square])


@njit
def is_castling(board_state, from_square, to_square):
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

        return diff > 1 or bool(board_state.rooks & board_state.occupied_co[board_state.turn] & BB_SQUARES[to_square])
    return False


@njit
def _ep_skewered(board_state, king, capturer):
    """
    Handle the special case where the king would be in check, if the pawn and its capturer disappear from the rank.

    Vertical skewers of the captured pawn are not possible. (Pins on the capturer are not handled here.)
    """
    # only using as workaround, won't do this long term
    if board_state.turn:
        last_double = board_state.ep_square - 8
    else:
        last_double = board_state.ep_square + 8

    occupancy = (board_state.occupied & ~BB_SQUARES[last_double] & ~BB_SQUARES[capturer] | BB_SQUARES[
        board_state.ep_square])

    # Horizontal attack on the fifth or fourth rank.
    horizontal_attackers = board_state.occupied_co[1 ^ board_state.turn] & (
                board_state.rooks | board_state.queens)
    if RANK_ATTACK_ARRAY[
        king, khash_get(RANK_ATTACK_INDEX_LOOKUP_TABLE, BB_RANK_MASKS[king] & occupancy, 0)] & horizontal_attackers:
        return True

    return False

@njit
def is_safe(board_state, king, blockers, from_square, to_square):
    if from_square == king:
        if is_castling(board_state, from_square, to_square):
            return True
        else:
            return not bool(_attackers_mask(board_state, 1 ^ board_state.turn, to_square, board_state.occupied))

    elif is_en_passant(board_state, from_square, to_square):
        return (pin_mask(board_state, board_state.turn, from_square) & BB_SQUARES[to_square] and
                not _ep_skewered(board_state, king, from_square))
    else:
        return not blockers & BB_SQUARES[from_square] or BB_RAYS[from_square, to_square] & BB_SQUARES[king]


@njit
def _slider_blockers(board_state, king):
    snipers = (((board_state.rooks | board_state.queens) &
               (RANK_ATTACK_ARRAY[king, 0] | FILE_ATTACK_ARRAY[king, 0])) |
               (DIAG_ATTACK_ARRAY[king, 0] & (board_state.bishops | board_state.queens)))

    blockers = 0
    for sniper in scan_reversed(snipers & board_state.occupied_co[1 ^ board_state.turn]):
        b = BB_BETWEEN[king, sniper] & board_state.occupied

        # Add to blockers if exactly one piece in between.
        if b and BB_SQUARES[msb(b)] == b:
            blockers |= b

    return blockers & board_state.occupied_co[board_state.turn]


@njit
def _attacked_for_king(board_state, path, occupied):
    for sq in scan_reversed(path):
        if _attackers_mask(board_state, 1 ^ board_state.turn, sq, occupied):
            return True
    return False


@njit
def _castling_uncovers_rank_attack(board_state, rook_bb, king_to):
    """
    Test the special case where we castle and our rook shielded us from
    an attack, so castling would be into check.
    """
    rank_pieces = BB_RANK_MASKS[king_to] & (board_state.occupied ^ rook_bb)
    sliders = (board_state.queens | board_state.rooks) & board_state.occupied_co[1 ^ board_state.turn]
    return RANK_ATTACK_ARRAY[king_to, khash_get(RANK_ATTACK_INDEX_LOOKUP_TABLE, rank_pieces, 0)] & sliders


@njit
def _from_chess960_tuple(board_state, from_square, to_square):
    if from_square == E1 and board_state.kings & BB_E1:
        if to_square == H1:
            return E1, G1, NO_PROMOTION_VALUE
        elif to_square == A1:
            return E1, C1, NO_PROMOTION_VALUE
    elif from_square == E8 and board_state.kings & BB_E8:
        if to_square == H8:
            return E8, G8, NO_PROMOTION_VALUE
        elif to_square == A8:
            return E8, C8, NO_PROMOTION_VALUE

    # promotion is set to 0 because this function only gets called when looking for castling moves,
    # which can't have promotions
    return from_square, to_square, NO_PROMOTION_VALUE


def pseudo_legal_ep_fn_creator(has_legal_move_checker=False):
    def set_pseudo_legal_ep(board_state, from_mask=BB_ALL, to_mask=BB_ALL, king=0, blockers=0):
        if not board_state['ep_square']:
            return False

        if not BB_SQUARES[board_state['ep_square']] & to_mask:
            return False

        if BB_SQUARES[board_state['ep_square']] & board_state['occupied']:
            return False

        capturers = (board_state['pawns'] & board_state['occupied_co'][board_state['turn']] & from_mask &
                     BB_PAWN_ATTACKS[1 ^  board_state['turn'], board_state['ep_square']] &
                     BB_RANKS[3 + board_state['turn']])

        if has_legal_move_checker:
            for capturer in scan_reversed(capturers):
                if is_safe(board_state, king, blockers, capturer, board_state['ep_square']):
                    return True
            return False
        else:
            for capturer in scan_reversed(capturers):
                board_state['unexplored_moves'][board_state.children_left] = np.array(
                    [capturer, board_state['ep_square'], 0],
                    dtype=np.uint8)
                board_state['children_left'] += 1

    return njit(set_pseudo_legal_ep)


set_pseudo_legal_ep = pseudo_legal_ep_fn_creator()
has_pseudo_legal_ep = pseudo_legal_ep_fn_creator(True)


def castling_fn_creator(has_legal_move_checker=False):
    def set_castling_moves(board_state, from_mask=BB_ALL, to_mask=BB_ALL, blockers=0):
        backrank = BB_RANK_1 if board_state['turn'] else BB_RANK_8

        king = board_state['occupied_co'][board_state['turn']] & board_state['kings'] & backrank & from_mask

        king &= -king #I think this can be removed

        if not king or _attacked_for_king(board_state, king, board_state['occupied']):
            return False

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

            if has_legal_move_checker:
                if not ((board_state['occupied'] ^ king ^ rook) & (empty_for_king | empty_for_rook) or
                        _attacked_for_king(board_state, empty_for_king, board_state['occupied'] ^ king) or
                        _castling_uncovers_rank_attack(board_state, rook, king_to)):
                    temp_tup = _from_chess960_tuple(board_state, msb(king), candidate)
                    if is_safe(board_state, king, blockers, temp_tup[0], temp_tup[1]):
                        return True
            else:
                if not ((board_state['occupied'] ^ king ^ rook) & (empty_for_king | empty_for_rook) or
                        _attacked_for_king(board_state, empty_for_king, board_state['occupied'] ^ king) or
                        _castling_uncovers_rank_attack(board_state, rook, king_to)):
                    board_state['unexplored_moves'][board_state.children_left, :] = _from_chess960_tuple(board_state,
                                                                                                         msb(king),
                                                                                                         candidate)
                    board_state['children_left'] += 1
        return False

    return njit(set_castling_moves)


set_castling_moves = castling_fn_creator()
has_castling_move = castling_fn_creator(True)


def pseudo_legal_move_fn_creator(has_legal_move_checker=False):
    def set_pseudo_legal_moves(board_state, from_mask=BB_ALL, to_mask=BB_ALL, king=0, blockers=0):
        """
        NOTES:
        1) All of the NumPy array creation when setting the unexplored_moves may or may not have speed penalties,
        but either way it would be much better to somehow just convince Numba that the values have
        dtype np.uint8 (which they do!).
        """
        cur_turn_occupied = board_state['occupied_co'][board_state['turn']]
        opponent_occupied = board_state['occupied_co'][1 ^ board_state['turn']]

        # Generate piece moves.
        non_pawns = cur_turn_occupied & ~board_state['pawns'] & from_mask

        for from_square in scan_reversed(non_pawns):

            moves = attacks_mask(board_state, from_square) & ~cur_turn_occupied & to_mask
            if has_legal_move_checker:
                for to_square in scan_reversed(moves):
                    if is_safe(board_state, king, blockers, from_square, to_square):
                        return True
            else:
                for to_square in scan_reversed(moves):
                    board_state['unexplored_moves'][board_state.children_left, :] = np.array([from_square, to_square, 0],
                                                                                             dtype=np.uint8)
                    board_state['children_left'] += 1

        # Generate castling moves.
        if from_mask & board_state['kings']:
            if has_legal_move_checker:
                if has_castling_move(board_state, from_mask, to_mask, blockers=blockers):
                    return True
            else:
                set_castling_moves(board_state, from_mask, to_mask)

        # The remaining moves are all pawn moves.
        pawns = board_state['pawns'] & cur_turn_occupied & from_mask
        if not pawns:
            return False

        # Generate pawn captures.
        capturers = pawns

        if has_legal_move_checker:
            for from_square in scan_reversed(capturers):
                targets = BB_PAWN_ATTACKS[board_state['turn'], from_square] & opponent_occupied & to_mask

                for to_square in scan_reversed(targets):
                    if is_safe(board_state, king, blockers, from_square, to_square):
                        return True
        else:
            for from_square in scan_reversed(capturers):
                targets = BB_PAWN_ATTACKS[board_state['turn'], from_square] & opponent_occupied & to_mask

                for to_square in scan_reversed(targets):
                    if square_rank(to_square) in [0, 7]:
                        board_state['unexplored_moves'][board_state.children_left:board_state.children_left + 4, 0] = from_square
                        board_state['unexplored_moves'][board_state.children_left:board_state.children_left + 4, 1] = to_square
                        board_state['unexplored_moves'][board_state.children_left:board_state.children_left + 4, 2] = (QUEEN, ROOK, BISHOP, KNIGHT)
                        board_state['children_left'] += 4
                    else:
                        board_state['unexplored_moves'][board_state.children_left, :] = np.array(
                            [from_square, to_square, 0],
                            dtype=np.uint8)
                        board_state['children_left'] += 1

        # Prepare pawn advance generation.
        if board_state['turn']:
            single_moves = pawns << 8 & ~board_state['occupied']
            double_moves = single_moves << 8 & ~board_state['occupied'] & (BB_RANK_3 | BB_RANK_4)
        else:
            single_moves = pawns >> 8 & ~board_state['occupied']
            double_moves = single_moves >> 8 & ~board_state['occupied'] & (BB_RANK_6 | BB_RANK_5)

        single_moves &= to_mask
        double_moves &= to_mask

        if has_legal_move_checker:
            for to_square in scan_reversed(single_moves):
                if not board_state['turn']:
                    from_square = to_square + 8
                else:
                    from_square = to_square - 8
                if is_safe(board_state, king, blockers, from_square, to_square):
                    return True

            for to_square in scan_reversed(double_moves):
                if not board_state['turn']:
                    from_square = to_square + 16
                else:
                    from_square = to_square - 16

                if is_safe(board_state, king, blockers, from_square, to_square):
                    return True

        else:
            # Generate single pawn moves.
            for to_square in scan_reversed(single_moves):
                if not board_state['turn']:
                    from_square = to_square + 8
                else:
                    from_square = to_square - 8

                if square_rank(to_square) in [0, 7]:
                    board_state['unexplored_moves'][board_state.children_left:board_state.children_left + 4, 0] = from_square
                    board_state['unexplored_moves'][board_state.children_left:board_state.children_left + 4, 1] = to_square
                    board_state['unexplored_moves'][board_state.children_left:board_state.children_left + 4, 2] = (QUEEN, ROOK, BISHOP, KNIGHT)
                    board_state['children_left'] += 4
                else:
                    board_state['unexplored_moves'][board_state.children_left, :] = np.array([from_square, to_square, 0],
                                                                                             dtype=np.uint8)
                    board_state['children_left'] += 1

            # Generate double pawn moves.
            for to_square in scan_reversed(double_moves):
                if not board_state['turn']:
                    from_square = to_square + 16
                else:
                    from_square = to_square - 16

                board_state['unexplored_moves'][board_state.children_left, :] = np.array([from_square, to_square, 0],
                                                                                         dtype=np.uint8)
                board_state['children_left'] += 1


        if has_legal_move_checker:
            # Generate en passant captures.
            if board_state['ep_square']:
                return has_pseudo_legal_ep(board_state, from_mask, to_mask, king=king, blockers=blockers)
            return False
        else:
            if board_state['ep_square']:
                set_pseudo_legal_ep(board_state, from_mask, to_mask)


    return njit(set_pseudo_legal_moves)

set_pseudo_legal_moves = pseudo_legal_move_fn_creator()
has_pseudo_legal_move = pseudo_legal_move_fn_creator(True)


@njit
def is_pseudo_legal_ep(board_state, from_mask, to_mask):
    if not board_state['ep_square']:
        return False

    if not BB_SQUARES[board_state['ep_square']] & to_mask:
        return False

    if BB_SQUARES[board_state['ep_square']] & board_state['occupied']:
        return False

    if board_state['pawns'] & board_state['occupied_co'][board_state['turn']] & from_mask & BB_PAWN_ATTACKS[
        1 ^ board_state['turn'], board_state['ep_square']] & BB_RANKS[3 + board_state['turn']]:

        return True
    return False


@njit
def is_pseudo_legal_castling_move(board_state, from_mask, to_mask):
    backrank = BB_RANK_1 if board_state['turn'] else BB_RANK_8

    king = board_state['occupied_co'][board_state['turn']] & board_state['kings'] & backrank & from_mask

    king &= -king

    if not king or _attacked_for_king(board_state, king, board_state['occupied']):
        return False


    candidates = board_state['castling_rights'] & backrank & to_mask
    if candidates:
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
def is_pseudo_legal_move(board_state, move):
    cur_turn_occupied = board_state['occupied_co'][board_state['turn']]
    opponent_occupied = board_state['occupied_co'][1 ^ board_state['turn']]

    from_mask = BB_SQUARES[move[0]]
    to_mask = BB_SQUARES[move[1]]

    # Generate piece moves.
    non_pawns = cur_turn_occupied & ~board_state['pawns'] & from_mask
    if non_pawns:
        if attacks_mask(board_state, msb(non_pawns)) & ~cur_turn_occupied & to_mask:
            if not move[2]:
                return True

    # Generate castling moves.
    if from_mask & board_state['kings']:
        chess960_from_square, chess960_to_square, _ = _to_chess960_tuple(board_state, move)
        if chess960_to_square != move[1]:
            if is_pseudo_legal_castling_move(board_state, BB_SQUARES[chess960_from_square], BB_SQUARES[chess960_to_square]):
                if not move[2]:
                    return True

    # The remaining possible moves are all pawn moves.
    pawns = board_state['pawns'] & cur_turn_occupied & from_mask
    if not pawns:
        return False

    # Check pawn captures.
    if board_state['turn']:
        targets = BB_PAWN_ATTACKS[WHITE, msb(pawns)] & opponent_occupied & to_mask
        if targets:
            if square_rank(msb(targets)) in [0,7]:
                return not move[2] in [0, PAWN, KING]
            else:
                return move[2] == 0
    else:
        if BB_PAWN_ATTACKS[BLACK, msb(pawns)] & opponent_occupied & to_mask:
            return True


    # Check pawn advance generation.
    if board_state['turn']:
        single_moves = pawns << 8 & ~board_state['occupied']
        if single_moves & to_mask:
            if square_rank(move[1]) in [0,7]:
                if not move[2] in [0, PAWN, KING]:
                    return True
            else:
                if not move[2]:
                    return True
        if (single_moves << 8 & ~board_state['occupied'] & (BB_RANK_3 | BB_RANK_4)) & to_mask:
            if not move[2]:
                return True
    else:
        single_moves = pawns >> 8 & ~board_state['occupied']
        if single_moves & to_mask:
            if square_rank(move[1]) in [0,7]:
                if not move[2] in [0, PAWN, KING]:
                    return True
            else:
                if not move[2]:
                    return True
        if (single_moves >> 8 & ~board_state['occupied'] & (BB_RANK_6 | BB_RANK_5)) & to_mask:
            if not move[2]:
                return True

    # Generate en passant captures.
    if board_state['ep_square'] and not move[2]:
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
        if BB_KING_ATTACKS[king] & ~board_state.occupied_co[board_state.turn] & ~attacked & to_mask:
            return True

    checker = msb(checkers)
    if BB_SQUARES[checker] == checkers:
        # If it captures or blocks a single checker.
        return ~board_state.kings & from_mask and (BB_BETWEEN[king, checker] | checkers) & to_mask

    return False


def evasions_creator(has_legal_move_checker=False):
    def set_evasions(board_state, king, checkers, from_mask=BB_ALL, to_mask=BB_ALL, blockers=0):
        sliders = checkers & (board_state.bishops | board_state.rooks | board_state.queens)

        attacked = np.uint64(0)
        for checker in scan_reversed(sliders):
            attacked |= BB_RAYS[king, checker] & ~BB_SQUARES[checker]

        if BB_SQUARES[king] & from_mask:
            if has_legal_move_checker:
                for to_square in scan_reversed(BB_KING_ATTACKS[king] & ~board_state.occupied_co[board_state.turn] & ~attacked & to_mask):
                    if is_safe(board_state, king, blockers, king, to_square):
                        return True
            else:
                for to_square in scan_reversed(BB_KING_ATTACKS[king] & ~board_state.occupied_co[board_state.turn] & ~attacked & to_mask):
                    board_state['unexplored_moves'][board_state.children_left, 0] = king
                    board_state['unexplored_moves'][board_state.children_left, 1] = to_square
                    board_state['unexplored_moves'][board_state.children_left, 2] = 0
                    board_state['children_left'] += 1

        checker = msb(checkers)
        if BB_SQUARES[checker] == checkers:
            # Capture or block a single checker.
            target = BB_BETWEEN[king][checker] | checkers
            if has_legal_move_checker:
                if has_pseudo_legal_move(board_state, ~board_state.kings & from_mask, target & to_mask, king, blockers):
                    return True
            else:
                set_pseudo_legal_moves(board_state, ~board_state.kings & from_mask, target & to_mask)

            # Capture the checking pawn en passant (but avoid yielding duplicate moves).
            if board_state.ep_square:
                if not BB_SQUARES[board_state.ep_square] & target:
                    if not board_state.turn:
                        last_double = board_state.ep_square + 8
                    else:
                        last_double = board_state.ep_square - 8
                    if last_double == checker:
                        if has_legal_move_checker:
                            return has_pseudo_legal_ep(board_state, from_mask, to_mask, king, blockers)
                        else:
                            set_pseudo_legal_ep(board_state, from_mask, to_mask)
        return False

    return njit(set_evasions)


set_evasions = evasions_creator()
has_evasion = evasions_creator(True)


@njit
def set_up_move_array(board_struct):
    king = msb(board_struct['kings'] & board_struct['occupied_co'][board_struct.turn])

    blockers = _slider_blockers(board_struct, king)
    checkers = _attackers_mask(board_struct, 1 ^ board_struct['turn'], king, board_struct['occupied'])

    # If in check
    if checkers:
        set_evasions(board_struct, king, checkers, BB_ALL, BB_ALL)
    else:
        set_pseudo_legal_moves(board_struct, BB_ALL, BB_ALL)

    legal_move_index = 0
    for j in range(board_struct['children_left']):
        if is_safe(board_struct, king, blockers, board_struct['unexplored_moves'][j, 0], board_struct['unexplored_moves'][j, 1]):
            board_struct['unexplored_moves'][legal_move_index] = board_struct['unexplored_moves'][j]
            legal_move_index += 1

    board_struct['unexplored_moves'][legal_move_index:board_struct['children_left'],:] = 255
    board_struct['children_left'] = legal_move_index

    if not board_struct['children_left']:
        board_struct['terminated'] = True
        board_struct['best_value'] = LOSS_RESULT_SCORES[board_struct['depth']] if checkers else TIE_RESULT_SCORE

@njit
def set_up_move_arrays(structs):
    for j in range(len(structs)):
        set_up_move_array(structs[j])

@njit
def has_legal_move(board_struct):
    king = msb(board_struct['kings'] & board_struct['occupied_co'][board_struct.turn])

    blockers = _slider_blockers(board_struct, king)
    checkers = _attackers_mask(board_struct, 1 ^ board_struct['turn'], king, board_struct['occupied'])

    # If in check
    if checkers:
        if has_evasion(board_struct, king, checkers, BB_ALL, BB_ALL, blockers):
            return True
        board_struct['best_value'] = LOSS_RESULT_SCORES[board_struct['depth']]

    else:
        if has_pseudo_legal_move(board_struct, BB_ALL, BB_ALL, king, blockers):
            return True
        board_struct['best_value'] = TIE_RESULT_SCORE

    return False


@njit
def set_up_move_array_except_move(board_struct, move_to_avoid):
    king = msb(board_struct['kings'] & board_struct['occupied_co'][board_struct.turn])

    blockers = _slider_blockers(board_struct, king)
    checkers = _attackers_mask(board_struct, 1 ^ board_struct['turn'], king, board_struct['occupied'])

    # If in check
    if checkers:
        set_evasions(board_struct, king, checkers, BB_ALL, BB_ALL)
    else:
        set_pseudo_legal_moves(board_struct, BB_ALL, BB_ALL)

    legal_move_index = 0
    for j in range(board_struct['children_left']):
        if np.any(move_to_avoid != board_struct['unexplored_moves'][j]):
            if is_safe(board_struct, king, blockers, board_struct['unexplored_moves'][j, 0], board_struct['unexplored_moves'][j, 1]):
                board_struct['unexplored_moves'][legal_move_index] = board_struct['unexplored_moves'][j]
                legal_move_index += 1

    board_struct['unexplored_moves'][legal_move_index:board_struct['children_left'],:] = 255
    board_struct['children_left'] = legal_move_index


@njit
def is_into_check(board_scalar, from_square, to_square):
    """
    Checks if the given move would leave the king in check or put it into
    check. The move must be at least pseudo legal.  This function was adapted from the Python-Chess version of it.
    """
    king = msb(board_scalar.occupied_co[board_scalar.turn] & board_scalar.kings)

    checkers = _attackers_mask(board_scalar, 1 ^ board_scalar.turn, king, board_scalar.occupied)
    if checkers:
        return not is_evasion(board_scalar, king, checkers, BB_SQUARES[from_square], BB_SQUARES[to_square])

    return not is_safe(board_scalar, king, _slider_blockers(board_scalar, king), from_square, to_square)


@njit
def is_legal_move(board_scalar, move):
    return is_pseudo_legal_move(board_scalar, move) and not is_into_check(board_scalar, move[0], move[1])


@njit
def perft_test_move_gen_helper(struct_array):
    for j in range(len(struct_array)):
        king = msb(struct_array[j]['kings'] & struct_array[j]['occupied_co'][struct_array[j]['turn']])

        blockers = _slider_blockers(struct_array[j], king)
        checkers = _attackers_mask(struct_array[j], 1 ^ struct_array[j]['turn'], king, struct_array[j]['occupied'])

        if checkers:
            set_evasions(struct_array[j], king, checkers, BB_ALL, BB_ALL)
        else:
            set_pseudo_legal_moves(struct_array[j], BB_ALL, BB_ALL)

        legal_move_index = 0
        for i in range(struct_array[j]['children_left']):
            if is_safe(struct_array[j], king, blockers, struct_array[j]['unexplored_moves'][i, 0], struct_array[j]['unexplored_moves'][i, 1]):
                struct_array[j]['unexplored_moves'][legal_move_index] = struct_array[j]['unexplored_moves'][i]
                legal_move_index += 1

        struct_array[j]['unexplored_moves'][legal_move_index:struct_array[j]['children_left'], :] = 255
        struct_array[j]['children_left'] = legal_move_index


def perft_test(struct_array, depth, print_info=False):
    if print_info:
        print("Starting depth %d with %d nodes."%(depth, len(struct_array)))

    if not depth:
        return len(struct_array)

    struct_array['unexplored_moves'] = np.full_like(struct_array['unexplored_moves'], 255)
    struct_array['children_left'] = np.zeros_like(struct_array['children_left'])

    perft_test_move_gen_helper(struct_array)


    if depth == 1:
        return np.sum(struct_array['children_left'])

    legal_moves = struct_array['unexplored_moves'][struct_array['unexplored_moves'][..., 0] != 255]

    repeated_struct_array = np.repeat(struct_array, struct_array['children_left'])

    #Not sure if these actually provide a speed increase
    repeated_struct_array = np.ascontiguousarray(repeated_struct_array)
    legal_moves = np.ascontiguousarray(legal_moves)

    push_moves(repeated_struct_array, legal_moves)

    return perft_test(repeated_struct_array, depth - 1)