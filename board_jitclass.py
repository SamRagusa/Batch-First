from numba import uint64, boolean, uint8

from collections import OrderedDict


from numba_board import *


#I'm not sure why I'm being required to import these functions, given the import directly above, but I'll figure it out soon (haven't really tried yet)
from numba_board import _remove_piece_at, _set_piece_at, _attacked_for_king, _attackers_mask, _castling_uncovers_rank_attack,_slider_blockers



move_spec = OrderedDict()
move_spec["from_square"] = uint8
move_spec["to_square"] = uint8
move_spec["promotion"] = uint8


@nb.jitclass(move_spec)
class Move:
    """
    Represents a move from a square to a square and possibly the promotion
    piece type.
    """

    def __init__(self, from_square, to_square, promotion):
        self.from_square = from_square
        self.to_square = to_square
        self.promotion = promotion


@njit(boolean(Move.class_type.instance_type, Move.class_type.instance_type))
def move_objects_equal(move1, move2):
    """
    Checks if the two given move objects are equal.
    """
    if move1.from_square == move2.from_square and move1.to_square == move2.to_square and move1.promotion == move2.promotion:
        return True
    return False



@njit(Move.class_type.instance_type(uint8, uint8))
def create_move(from_square, to_square):
    """
    For use when not using promotions
    """
    return Move(from_square, to_square, np.uint8(0))



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
board_state_spec["ep_square"] = nb.optional(uint8)

board_state_spec["halfmove_clock"] = uint8

board_state_spec["cur_hash"] = uint64


@nb.jitclass(board_state_spec)
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






@njit(BoardState.class_type.instance_type(BoardState.class_type.instance_type))
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





@njit(boolean(BoardState.class_type.instance_type, Move.class_type.instance_type))
def is_zeroing(board_state, move):
    """
    Checks if the given pseudo-legal move is a capture or pawn move.
    """
    if board_state.turn:
        return np.bool_(
            BB_SQUARES[move.from_square] & board_state.pawns or BB_SQUARES[move.to_square] & board_state.occupied_b)
    return np.bool_(
        BB_SQUARES[move.from_square] & board_state.pawns or BB_SQUARES[move.to_square] & board_state.occupied_w)


@njit
def _from_chess960(board_state, from_square, to_square):
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


# This should likely not be used/needed long term
@njit(Move.class_type.instance_type(BoardState.class_type.instance_type, Move.class_type.instance_type))
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







@njit(uint64(BoardState.class_type.instance_type, Move.class_type.instance_type))
def push_with_hash_update(board_state, move):
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



@njit(BoardState.class_type.instance_type(BoardState.class_type.instance_type, Move.class_type.instance_type))
def copy_push(board_state, move):
    to_push_move = copy_board_state(board_state)
    push_with_hash_update(to_push_move, move)
    return to_push_move





@njit((BoardState.class_type.instance_type, uint64, uint64))
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
            BB_PAWN_ATTACKS[BLACK, temp_ep_square] & BB_RANKS[4])
    else:
        capturers = (
            board_state.pawns & board_state.occupied_b & from_mask &
            BB_PAWN_ATTACKS[WHITE, temp_ep_square] & BB_RANKS[3])

    for capturer in scan_reversed(capturers):
        yield create_move(capturer, temp_ep_square)

    return





@njit(boolean(BoardState.class_type.instance_type, Move.class_type.instance_type))
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



@njit
def generate_castling_moves(board_state, from_mask=BB_ALL, to_mask=BB_ALL):
    if board_state.turn:
        backrank = BB_RANK_1
        king = board_state.occupied_w & board_state.kings & backrank & from_mask
    else:
        backrank = BB_RANK_8
        king = board_state.occupied_b & board_state.kings & backrank & from_mask

    king = king & -king

    if not king or _attacked_for_king(board_state, king, board_state.occupied):
        return

    bb_c = BB_FILE_C & backrank
    bb_d = BB_FILE_D & backrank
    bb_f = BB_FILE_F & backrank
    bb_g = BB_FILE_G & backrank

    for candidate in scan_reversed(board_state.castling_rights & backrank & to_mask):
        rook = BB_SQUARES[candidate]

        empty_for_rook = np.uint64(0)
        empty_for_king = np.uint64(0)

        if rook < king:
            king_to = msb(bb_c)
            if not rook & bb_d:
                empty_for_rook = BB_BETWEEN[candidate, msb(bb_d)] | bb_d
            if not king & bb_c:
                empty_for_king = BB_BETWEEN[msb(king), king_to] | bb_c
        else:
            king_to = msb(bb_g)
            if not rook & bb_f:
                empty_for_rook = BB_BETWEEN[candidate, msb(bb_f)] | bb_f
            if not king & bb_g:
                empty_for_king = BB_BETWEEN[msb(king), king_to] | bb_g

        if not ((board_state.occupied ^ king ^ rook) & (empty_for_king | empty_for_rook) or
                    _attacked_for_king(board_state, empty_for_king, board_state.occupied ^ king) or
                    _castling_uncovers_rank_attack(board_state, rook, king_to)):
            yield _from_chess960(board_state, msb(king), candidate)

    return


@njit((BoardState.class_type.instance_type, uint64, uint64))
def generate_pseudo_legal_moves(board_state, from_mask=BB_ALL, to_mask=BB_ALL):
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

        moves = attacks_mask(board_state, from_square) & ~our_pieces & to_mask
        for to_square in scan_reversed(moves):
            yield create_move(from_square, to_square)

    # Generate castling moves.
    if from_mask & board_state.kings:
        for move in generate_castling_moves(board_state, from_mask, to_mask):
            yield move

    # The remaining moves are all pawn moves.
    pawns = board_state.pawns & cur_turn_occupied & from_mask
    if not pawns:
        return

    # Generate pawn captures.
    capturers = pawns
    for from_square in scan_reversed(capturers):
        if board_state.turn:
            targets = (BB_PAWN_ATTACKS[WHITE, from_square] &
                       opponent_occupied & to_mask)
        else:
            targets = (BB_PAWN_ATTACKS[BLACK, from_square] &
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



@njit((BoardState.class_type.instance_type, uint8, uint64, uint64, uint64))
def _generate_evasions(board_state, king, checkers, from_mask=BB_ALL, to_mask=BB_ALL):
    sliders = checkers & (board_state.bishops | board_state.rooks | board_state.queens)

    attacked = np.uint64(0)
    for checker in scan_reversed(sliders):
        attacked |= BB_RAYS[king, checker] & ~BB_SQUARES[checker]

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
        target = BB_BETWEEN[king, checker] | checkers


        for move in generate_pseudo_legal_moves(board_state, ~board_state.kings & from_mask, target & to_mask):
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


@njit
def is_in_check(board_state):
    if board_state.turn == TURN_WHITE:
        king = msb(board_state.occupied_w & board_state.kings)
        return bool(_attackers_mask(board_state, TURN_BLACK, king, board_state.occupied))
    else:
        king = msb(board_state.occupied_b & board_state.kings)
        return bool(_attackers_mask(board_state, TURN_WHITE, king, board_state.occupied))


@njit(boolean(BoardState.class_type.instance_type, Move.class_type.instance_type))
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

    return False


@njit(boolean(BoardState.class_type.instance_type, uint8, uint8))
def _ep_skewered(board_state, king, capturer):
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
        if RANK_ATTACK_ARRAY[king, khash_get(RANK_ATTACK_INDEX_LOOKUP_TABLE, BB_RANK_MASKS[king] & occupancy, 0)] & horizontal_attackers:
            return True

    else:
        last_double = temp_ep_square + 8
        occupancy = (board_state.occupied & ~BB_SQUARES[last_double] & ~BB_SQUARES[capturer] | BB_SQUARES[temp_ep_square])

        # Horizontal attack on the fifth or fourth rank.
        horizontal_attackers = board_state.occupied_w & (board_state.rooks | board_state.queens)
        if RANK_ATTACK_ARRAY[king,
                             khash_get(
                                 RANK_ATTACK_INDEX_LOOKUP_TABLE,
                                 BB_RANK_MASKS[king] & occupancy,
                                 0)] & horizontal_attackers:
            return True

    return False




@njit(boolean(BoardState.class_type.instance_type, uint8, uint64, Move.class_type.instance_type))
def _is_safe(board_state, king, blockers, move):
    if move.from_square == king:
        if is_castling(board_state, move):
            return True
        else:
            return not bool(_attackers_mask(board_state, not board_state.turn, move.to_square, board_state.occupied))

    elif is_en_passant(board_state, move):
        return (pin_mask(board_state, board_state.turn, move.from_square) & BB_SQUARES[move.to_square] and
                not _ep_skewered(board_state, king, move.from_square))
    else:
        return (not blockers & BB_SQUARES[move.from_square] or
                BB_RAYS[move.from_square, move.to_square] & BB_SQUARES[king])



@njit((BoardState.class_type.instance_type, uint64, uint64), nogil=True)
def generate_legal_moves(board_state, from_mask=BB_ALL, to_mask=BB_ALL):
    if board_state.turn == TURN_WHITE:
        king = msb(board_state.kings & board_state.occupied_w)
    else:
        king = msb(board_state.kings & board_state.occupied_b)
    blockers = _slider_blockers(board_state, king)
    checkers = _attackers_mask(board_state, not board_state.turn, king, board_state.occupied)
    #If in check
    if checkers:#If no moves are found it needs to be passed along that the board is in check, so it doesn't need to be computed again directly afterwards
        for move in _generate_evasions(board_state, king, checkers, from_mask, to_mask):
            if _is_safe(board_state, king, blockers, move):
                yield move
    else:#If no moves are found it needs to be passed along that the board is not in check, so it doesn't need to be computed again directly afterwards
        for move in generate_pseudo_legal_moves(board_state, from_mask, to_mask):
            if _is_safe(board_state, king, blockers, move):
                yield move




@njit(boolean(BoardState.class_type.instance_type), nogil=True)
def has_legal_move(board_state):
    """
    Checks if there exists a legal move
    """
    return any(generate_legal_moves(board_state, BB_ALL, BB_ALL))

def generate_move_to_enumeration_dict():
    """
    Generates a dictionary where the keys are (from_square, to_square) and their values are the move number
    that move has been assigned.  It is done in a way such that for move number N from board B, if you were to flip B
    vertically, the same move would have number 1792-N. (there are 1792 moves recognized)


    IMPORTANT NOTES:
    1) This ignores the fact that not all pawn promotions are the same, this effects the number of logits
    in the move scoring ANN
    2) This will be phased out as the ANNs in use start to use the new mapping in engine_constants.py
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

    switch_square_fn = lambda x : x ^ 0x38

    total_possible_moves = len(possible_moves)*2 - 1

    for (from_square, to_square), move_num in list(possible_moves.items()):
        possible_moves[switch_square_fn(from_square), switch_square_fn(to_square)] = total_possible_moves - move_num

    return possible_moves







@njit(uint64(BoardState.class_type.instance_type, uint8))
def traditional_perft_test(board_state, depth):
    if depth == 0:
        return np.uint64(1)

    num_nodes = np.uint64(0)

    for move in generate_legal_moves(board_state, BB_ALL, BB_ALL):
        num_nodes += traditional_perft_test(copy_push(board_state, move), depth - 1)

    return num_nodes


