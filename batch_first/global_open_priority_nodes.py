from .classes_and_structs import *


@njit
def should_not_terminate(game_node):
    cur_node = game_node
    while cur_node is not None:
        if cur_node.struct.terminated:
            return False
        cur_node = cur_node.parent
    return True


@njit
def append_non_terminating(to_check, root):
    while not to_check is None:
        if should_not_terminate(to_check.held_node):
            root.next_holder = to_check
            root = root.next_holder

        to_check = to_check.next_holder

    root.next_holder = None
    return root


@njit
def append_non_terminating_with_counting(to_check, root, max_to_get):
    num_found = 0
    while not to_check is None:
        if should_not_terminate(to_check.held_node):
            root.next_holder = to_check
            root = root.next_holder

            num_found += 1
            if num_found == max_to_get:
                to_check = to_check.next_holder
                break

        to_check = to_check.next_holder

    root.next_holder = None
    return root, to_check, num_found



class GlobalNodeList(object):
    def is_empty(self):
        """
        Checks if the node list is empty.

        :return: A boolean value indicating if the list is empty.
        """
        raise NotImplementedError("This method must be implemented!")

    def insert_nodes_and_get_next_batch(self, to_insert, scores):
        """
        Inserts the given nodes into the stored list, and gets the next batch of nodes to be computed.

        :param to_insert: An ndarray of the nodes to be inserted into the list
        :param scores: An ndarray of the values used for prioritization, corresponding to the given nodes to insert
        :return: An ndarray of the nodes to be computed in the next batch
        """
        raise NotImplementedError("This method must be implemented!")

    def clear_list(self):
        """
        Clears the list so that it is empty.
        """
        raise NotImplementedError("This method must be implemented!")



@njit
def insert_nodes(bins, bin_lls, bin_lengths, non_empty_mask, to_insert, scores, zero_shift):
    scores -= zero_shift
    scores = np.abs(scores)

    bin_indices = np.digitize(scores, bins)
    for j in range(len(scores)):
        temp_next = to_insert.next_holder

        if non_empty_mask[bin_indices[j]]:
            to_insert.next_holder = bin_lls[bin_indices[j]]
        else:
            to_insert.next_holder = None
            non_empty_mask[bin_indices[j]] = True

        bin_lengths[bin_indices[j]] += 1
        bin_lls[bin_indices[j]] = to_insert
        to_insert = temp_next


@njit
def get_batch(bin_lls, bin_lengths, non_empty_mask, max_batch_size_to_accept):
    dummy_root = create_dummy_node_holder()
    end_node = dummy_root
    num_found = 0

    temp_dummy_node = create_dummy_node_holder()
    for bin_index in np.where(non_empty_mask)[0]:
        end_node, bin_leftover, just_found = append_non_terminating_with_counting(
            bin_lls[bin_index], end_node, max_batch_size_to_accept - num_found)

        num_found += just_found

        if not bin_leftover is None:
            bin_lls[bin_index] = bin_leftover
            non_empty_mask[bin_index] = True
            bin_lengths[bin_index] = len_node_holder(bin_leftover)
            break

        bin_lls[bin_index] = temp_dummy_node
        non_empty_mask[bin_index] = False
        bin_lengths[bin_index] = 0

        if num_found == max_batch_size_to_accept:
            break

    return dummy_root.next_holder


@njit
def pop_all_non_terminating(bin_lls, bin_lengths, non_empty_mask):
    """
    Set all the bin arrays to empty (by use of a mask), and return an array of all the nodes currently
    in a bin array that should not terminate.
    """
    dummy_root = create_dummy_node_holder()
    end_ll_node = dummy_root

    temp_dummy_node = create_dummy_node_holder()
    for bin_index in np.where(non_empty_mask)[0]:
        end_ll_node = append_non_terminating(bin_lls[bin_index], end_ll_node)
        bin_lls[bin_index] = temp_dummy_node

    bin_lengths[non_empty_mask] = 0
    non_empty_mask[non_empty_mask] = False

    return dummy_root.next_holder, end_ll_node


@njit
def insert_and_get_batch(to_insert, scores, bins, bin_ll_holder, bin_lengths, non_empty_mask, max_batch_size_to_accept, zero_shift):
    own_len = np.sum(bin_lengths)

    # This should not be using self.max_batch_size_to_accept for the initial check (here), instead should probably
    # be using a value greater than that, because even if 0 nodes are terminating, the time saved will likely
    # be more than the time spent computing the extra nodes (though it's unlikely that 0 nodes will be terminated
    # in actual play)
    if len(scores) + own_len < max_batch_size_to_accept:
        if own_len == 0:
            dummy_root = create_dummy_node_holder()
            append_non_terminating(to_insert, dummy_root)
            return dummy_root.next_holder
        elif len(scores) == 0:
            bin_lls = get_list_from_holder_holder(bin_ll_holder)
            to_return = pop_all_non_terminating(bin_lls, bin_lengths, non_empty_mask)[0]
        else:
            bin_lls = get_list_from_holder_holder(bin_ll_holder)
            to_return, end_node = pop_all_non_terminating(bin_lls, bin_lengths, non_empty_mask)
            end_node.next_holder = to_insert

        clear_holder_holder(bin_ll_holder)
        return to_return

    bin_lls = get_list_from_holder_holder(bin_ll_holder)

    insert_nodes(bins, bin_lls, bin_lengths, non_empty_mask, to_insert, scores, zero_shift)

    batch_to_return = get_batch(bin_lls, bin_lengths, non_empty_mask, max_batch_size_to_accept)

    new_holder_holder = get_holder_holder_from_list(bin_lls)
    bin_ll_holder.held = new_holder_holder.held
    bin_ll_holder.next = new_holder_holder.next

    return batch_to_return



class PriorityBins(GlobalNodeList):
    def __init__(self, bins, max_batch_size_to_accept, zero_shift=0, save_info=False):
        num_bins = (len(bins) + 1)

        self.bins = bins[::-1]

        self.bin_lengths = np.zeros(num_bins, dtype=np.int32)
        self.non_empty_mask = np.zeros(num_bins, dtype=np.bool_)

        temp_dummy_node = create_dummy_node_holder()
        self.holder_holder = get_holder_holder_from_list([temp_dummy_node for _ in range(num_bins)])

        self.max_batch_size_to_accept = max_batch_size_to_accept
        self.zero_shift = zero_shift

        self.save_info = save_info

        if save_info:
            self.reset_logs()

    def reset_logs(self):
        self.total_in = 0
        self.total_out = 0

    def __len__(self):
        return np.sum(self.bin_lengths)

    def is_empty(self):
        return not np.any(self.non_empty_mask)

    def num_non_empty(self):
        return np.sum(self.non_empty_mask)

    def largest_bin(self):
        return np.max(self.bin_lengths)

    def clear_list(self):
        clear_holder_holder(self.holder_holder)

        self.bin_lengths[self.non_empty_mask] = 0
        self.non_empty_mask[self.non_empty_mask] = False

    def insert_nodes_and_get_next_batch(self, to_insert, scores):
        if self.save_info:
            self.total_in += len(scores)

        to_return = insert_and_get_batch(
            to_insert,
            scores,
            self.bins,
            self.holder_holder,
            self.bin_lengths,
            self.non_empty_mask,
            self.max_batch_size_to_accept,
            self.zero_shift)

        if self.save_info and to_return:
            self.total_out += len_node_holder(to_return)

        return to_return

