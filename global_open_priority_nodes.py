import numpy as np
from numba import njit


@njit(nogil=True)
def should_not_terminate(game_node):
    cur_node = game_node
    while cur_node is not None:
        if cur_node.terminated:
            # This is here so that if a thread is stopped halfway through looking through a set of nodes, the next
            # time this function is called on the same node, it will complete much faster.  It's not crucial to the algorithm.
            game_node.terminated = True

            return False
        cur_node = cur_node.parent
    return True



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




class PriorityBins:
    def __init__(self, bins, min_batch_size_to_accept, testing=False):
        self.bins = bins

        #This should very likely be a 2d numpy array and not start with size 0, as it would allow for faster
        self.bin_arrays = [np.array([], dtype=np.object) for _ in range(len(bins)+1)]
        self.min_batch_size_to_accept = min_batch_size_to_accept

        self.non_empty_mask = np.zeros([len(self.bin_arrays)],dtype=np.bool_)
        self.testing = testing


    def __len__(self):
        answer = 0
        for bin, non_empty in zip(self.bin_arrays, self.non_empty_mask):
            if non_empty:
                answer += len(bin)
        return answer


    def is_empty(self):
        return not np.any(self.non_empty_mask)


    def largest_bin(self):
        if np.any(self.non_empty_mask):
            return max((len(self.bin_arrays[index]) for index in np.arange(len(self.bin_arrays))[self.non_empty_mask]))
        return 0


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


    def insert_nodes_and_get_next_batch(self, to_insert):
        bin_indices = np.digitize(np.array([node.next_move_score for node in to_insert], dtype=np.float32), self.bins)

        if len(to_insert) != 0:
            bins_to_insert = split_by_bins(to_insert, bin_indices)
        else:
            bins_to_insert = []

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

            not_terminated_mask = np.array(list(map(should_not_terminate, bin_to_look_at)), np.bool_)
            nodes_chosen += np.sum(not_terminated_mask)
            for_completion.append((bin_to_look_at, not_terminated_mask)) #Make sure these are veiws not copys

            if nodes_chosen >= self.min_batch_size_to_accept:
                # Insert every node not used in the next batch
                for bin_index, for_insertion in bins_to_insert[last_bin_index_used_in_batch + 1:]:
                    if self.non_empty_mask[bin_index]:
                        self.bin_arrays[bin_index] = np.append(self.bin_arrays[bin_index], for_insertion)
                    else:
                        self.bin_arrays[bin_index] = for_insertion
                        self.non_empty_mask[bin_index] = True

                break

        return np.concatenate([ary[mask] for ary, mask in for_completion])

