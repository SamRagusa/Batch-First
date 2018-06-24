import numpy as np
import numba as nb



@nb.njit
def should_not_terminate(game_node):
    cur_node = game_node
    while cur_node is not None:
        if cur_node.board_struct[0].terminated:
            return False
        cur_node = cur_node.parent
    return True


def should_not_terminate_node_array(node_array):
    return list(map(should_not_terminate, node_array))



def should_not_terminate_node_array_with_counting(node_array, max_to_get):
    num_not_terminating = 0
    nodes_checked = len(node_array)
    should_not_terminate_mask = np.zeros(len(node_array), dtype=np.bool_)
    for j in range(len(node_array)):
        if should_not_terminate(node_array[j]):
            should_not_terminate_mask[j] = True
            num_not_terminating += 1

            if num_not_terminating == max_to_get:
                nodes_checked = j + 1
                should_not_terminate_mask = should_not_terminate_mask[:nodes_checked]
                break

    return should_not_terminate_mask, num_not_terminating, nodes_checked


def split_by_bins(to_insert, bin_indices):
    """
    :return: An array of tuples, the first element of which is a bin index, and the second is a numpy array of
    nodes which belong to the bin specified by the first element
    """
    sorted_indices = bin_indices.argsort()
    sorted_bin_indices = bin_indices[sorted_indices]
    cut_indices = np.flatnonzero(sorted_bin_indices[1:] != sorted_bin_indices[:-1])+1
    slice_things = np.r_[0, cut_indices, len(sorted_bin_indices)+1]
    out = [(sorted_bin_indices[i], to_insert[sorted_indices[i:j]]) for i,j in zip(slice_things[:-1], slice_things[1:])]
    return out





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



class PriorityBins(GlobalNodeList):
    def __init__(self, bins, max_batch_size_to_accept, testing=False):
        self.bins = bins

        self.bin_arrays = [np.array([], dtype=np.object) for _ in range(len(bins)+1)]
        self.max_batch_size_to_accept = max_batch_size_to_accept

        self.non_empty_mask = np.zeros([len(self.bin_arrays)],dtype=np.bool_)
        self.testing = testing

        self.temp_aranged_array = np.arange(len(self.bin_arrays))

    def __len__(self):
        return sum(len(self.bin_arrays[bin_index]) for bin_index in self.temp_aranged_array[self.non_empty_mask])

    def is_empty(self):
        return not np.any(self.non_empty_mask)

    def num_non_empty(self):
        return np.sum(self.non_empty_mask)

    def largest_bin(self):
        if np.any(self.non_empty_mask):
            return max((len(self.bin_arrays[index]) for index in np.arange(len(self.bin_arrays))[self.non_empty_mask]))
        return 0

    def clear_list(self):
        for index in self.temp_aranged_array[self.non_empty_mask]:
            self.bin_arrays[index] = np.array([], dtype=np.object)
        self.non_empty_mask = np.zeros([len(self.bin_arrays)],  dtype=np.bool_)


    def pop_all_non_terminating(self):
        """
        Set all the bin arrays to empty (by use of a mask), and return an array of all the nodes currently
        in a bin array that should not terminate.
        """
        to_return = np.concatenate([
            self.bin_arrays[j][
                should_not_terminate_node_array(
                    self.bin_arrays[j])] for j, should_concat in enumerate(self.non_empty_mask) if should_concat])

        self.non_empty_mask[:] = False
        return to_return

    def best_bin_iterator(self, bins_to_insert):
        """
        Iterate through the bins being given to insert, and the non-empty bins stored in priority order.  If there are
        non-empty stored bins where nodes given to insert belong, the nodes being given for insertion will be given
        before the bin stored (for a more depth-first oriented search).

        :return: A size two tuple, the first element is a bool saying if it's in the bins to insert or in the
        bins stored, the second element being the index in whichever array of bins it's referring.
        """
        non_empty_bins = np.arange(len(self.bin_arrays))[self.non_empty_mask]
        index_in_non_empty_bins = 0
        for index_in_sorted_bins, to_insert in enumerate(bins_to_insert):
            while index_in_non_empty_bins < len(non_empty_bins) and non_empty_bins[index_in_non_empty_bins] < to_insert[0]:
                yield (True, non_empty_bins[index_in_non_empty_bins])
                index_in_non_empty_bins += 1
            yield (False, index_in_sorted_bins)

        while index_in_non_empty_bins < len(non_empty_bins):
            yield (True, non_empty_bins[index_in_non_empty_bins])
            index_in_non_empty_bins += 1

    def insert_nodes_and_get_next_batch(self, to_insert, scores):
        """
        SPEED IMPROVEMENTS TO MAKE:
        1) At any point if the number of known not terminated nodes plus the number of possible non_terminating nodes
        is less than self.max_batch_size_to_accept, then default to returning all nodes(which shouldn't terminate)
        without regard to their order


        NOTES:
        1) Should consider checking if performance is impacted by taking only part of a bin.  The concern here is that
        after the next iteration, it could discover enough nodes with a high enough priority to hide the bin
        which was partially chosen, and potentially prevent better options from being explored sooner
        """
        own_len = len(self)

        # This should not be using self.max_batch_size_to_accept for the initial check (here), instead should probably
        # be using a value greater than that, because even if 0 nodes are terminating, the time saved will likely
        # be more than the time spent computing the extra nodes (though it's unlikely that 0 nodes will be terminated
        # in actual play)
        if len(to_insert) + own_len < self.max_batch_size_to_accept:
            if len(to_insert) == 0:
                return self.pop_all_non_terminating()

            if own_len == 0:
                return to_insert[should_not_terminate_node_array(to_insert)]

            return np.concatenate([to_insert[should_not_terminate_node_array(to_insert)], self.pop_all_non_terminating()])


        bin_indices = np.digitize(scores, self.bins)

        if len(to_insert) == 0:
            bins_to_insert = []
        else:
            bins_to_insert = split_by_bins(to_insert, bin_indices)

        last_bin_index_used_in_batch = -1
        for_completion = []
        nodes_chosen = 0
        for next_array_info in self.best_bin_iterator(bins_to_insert):
            if next_array_info[0]:
                bin_to_look_at = self.bin_arrays[next_array_info[1]]
                self.non_empty_mask[next_array_info[1]] = False
            else:
                bin_to_look_at = bins_to_insert[next_array_info[1]][1]
                last_bin_index_used_in_batch += 1


            not_terminated_mask, num_not_terminating, nodes_checked = should_not_terminate_node_array_with_counting(
                bin_to_look_at,
                self.max_batch_size_to_accept - nodes_chosen)

            nodes_chosen += num_not_terminating

            # If the maximum number of nodes was found before completing current array termination check
            if nodes_checked == len(bin_to_look_at):
                for_completion.append((bin_to_look_at, not_terminated_mask))
            else:

                for_completion.append((bin_to_look_at[:nodes_checked], not_terminated_mask))

                if next_array_info[0]:
                    self.bin_arrays[next_array_info[1]] = bin_to_look_at[nodes_checked:]
                    self.non_empty_mask[next_array_info[1]] = True
                else:#If the bin being looked at was supposed to be inserted
                    if self.non_empty_mask[next_array_info[1]]:
                        self.bin_arrays[next_array_info[1]] = np.append(
                            self.bin_arrays[next_array_info[1]],
                            bin_to_look_at[nodes_checked:])
                    else:
                        self.bin_arrays[next_array_info[1]] = bin_to_look_at[nodes_checked:]
                        self.non_empty_mask[next_array_info[1]] = True


            if nodes_chosen >= self.max_batch_size_to_accept:
                # Insert every node not used in the next batch
                for bin_index, for_insertion in bins_to_insert[last_bin_index_used_in_batch + 1:]:
                    if self.non_empty_mask[bin_index]:
                        self.bin_arrays[bin_index] = np.append(self.bin_arrays[bin_index], for_insertion)
                    else:
                        self.bin_arrays[bin_index] = for_insertion
                        self.non_empty_mask[bin_index] = True
                break

        return np.concatenate([ary[mask] for ary, mask in for_completion])
