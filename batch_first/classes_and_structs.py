from collections import OrderedDict

from . import *



numpy_node_info_dtype = np.dtype(
    [("pawns", np.uint64),
     ("knights", np.uint64),
     ("bishops", np.uint64),
     ("rooks", np.uint64),
     ("queens", np.uint64),
     ("kings", np.uint64),
     ("occupied_co", np.uint64, (2)),
     ("occupied", np.uint64),
     ("turn", np.int8),
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


def create_node_info_from_python_chess_board(board, depth=255, separator=0):
    return np.array(
        [(board.pawns,
          board.knights,
          board.bishops,
          board.rooks,board.queens,
          board.kings,
          board.occupied_co,
          board.occupied,
          np.int8(board.turn),
          board.castling_rights,
          board.ep_square if not board.ep_square is None else NO_EP_SQUARE,
          board.halfmove_clock,
          zobrist_hash(board),
          False,                # terminated
          separator,
          depth,
          MIN_FLOAT32_VAL,      # best_value
          np.full([MAX_MOVES_LOOKED_AT, 3], 255, dtype=np.uint8),                # unexplored moves
          np.full([MAX_MOVES_LOOKED_AT], MIN_FLOAT32_VAL, dtype=np.float32),     # unexplored move scores
          np.full([3], 255, dtype=np.uint8), # The move made to reach the position this board represents
          0,        # next_move_index  (the index in the stored moves where the next move to make is)
          0)],      # children_left (the number of children which have yet to returne a value, or be created)
        dtype=numpy_node_info_dtype)


def create_node_info_from_fen(fen, depth, separator):
    return create_node_info_from_python_chess_board(chess.Board(fen), depth, separator)



game_node_type = nb.deferred_type()

gamenode_spec = OrderedDict()

gamenode_spec["board_struct"] = numba_node_info_type[:]
gamenode_spec["parent"] = nb.optional(game_node_type)




@nb.jitclass(gamenode_spec)
class GameNode:
    def __init__(self, board_struct, parent):
        self.board_struct = board_struct

        # Eventually this should be some sort of list, so that updating multiple parents is possible
        # when handling transpositions which are open at the same time.
        self.parent = parent

    @property
    def struct(self):
        return self.board_struct[0]


game_node_type.define(GameNode.class_type.instance_type)



game_node_holder_type = nb.deferred_type()

game_node_holder_spec = OrderedDict()

game_node_holder_spec["held_node"] = GameNode.class_type.instance_type
game_node_holder_spec["next_holder"] = nb.optional(game_node_holder_type)

@nb.jitclass(game_node_holder_spec)
class GameNodeHolder:
    """
    A jitclass used for representing a linked list of GameNode objects.  (this is mainly used for compilation purposes)

    NOTES:
    1) This shouldn't be needed at all and a 'next' SOMETHING should just be added to the GameNode class, but
    Numba won't let that happen (yet)
    """
    def __init__(self, held_node, next_holder):
        self.held_node = held_node
        self.next_holder = next_holder

    @property
    def struct(self):
        return self.held_node.struct

game_node_holder_type.define(GameNodeHolder.class_type.instance_type)



game_node_holder_holder_type = nb.deferred_type()

game_node_holder_holder_spec = OrderedDict()

game_node_holder_holder_spec["held"] = GameNodeHolder.class_type.instance_type
game_node_holder_holder_spec["next"] = nb.optional(game_node_holder_holder_type)

@nb.jitclass(game_node_holder_holder_spec)
class GameNodeHolderHolder:
    """
    This is a temporary class used to avoid the time required for Numba's boxing and unboxing when using Lists of
    JitClass objects.  It will be removed when more JIT coverage allows.

    """
    def __init__(self, held, next):
        self.held = held
        self.next = next

game_node_holder_holder_type.define(GameNodeHolderHolder.class_type.instance_type)



@njit
def get_list_from_holder_holder(holder):
    to_return = []
    while not holder is None:
        to_return.append(holder.held)
        holder = holder.next
    return to_return

@njit
def get_holder_holder_from_list(lst):
    next_holder = None
    for sub_holder in lst[::-1]:
        next_holder = GameNodeHolderHolder(sub_holder, next_holder)
    return next_holder

@njit
def clear_holder_holder(holder):
    dummy_sub_holder = create_dummy_node_holder()
    while not holder is None:
        holder.held = dummy_sub_holder
        holder = holder.next



@njit
def len_node_holder(ll):
    count = 0
    while not ll is None:
        count += 1
        ll = ll.next_holder
    return count


@njit
def create_dummy_node_holder():
    return GameNodeHolder(GameNode(np.empty(1, numpy_node_info_dtype), None), None)


@njit
def filter_holders_then_append(root, holder, mask, append_to_end):
    if holder is None:
        root.next_holder = append_to_end
        return 0

    root.next_holder = holder

    holder = root
    mask_index = 0
    total_kept = 0
    while not holder.next_holder is None:
        if mask[mask_index]:
            holder = holder.next_holder
            total_kept += 1
        else:
            holder.next_holder = holder.next_holder.next_holder
        mask_index += 1

    holder.next_holder = append_to_end
    return total_kept
