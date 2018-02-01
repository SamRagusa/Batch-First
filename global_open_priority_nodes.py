import numpy as np




def terminate_if_possible(game_node):
    if game_node.terminated:
        return True

    if not game_node.parent is None:
        if terminate_if_possible(game_node.parent):
            game_node.terminated = True
            game_node.parent.children_left -= 1
            return True
    return False



class PriorityBins:
    def __init__(self, bins, max_batch_size_to_check):
        self.bins = bins
        self.bin_arrays = [np.array([], dtype=np.object) for _ in range(len(bins)+1)]
        self.max_batch_size_to_check = max_batch_size_to_check


    def __len__(self):
        total_len = 0
        for cur_array in self.bin_arrays:
            total_len += len(cur_array)
        return total_len


    def largest_bin(self):
        return max([len(cur_array) for cur_array in self.bin_arrays])


    # @profile
    def insert_batch_and_get_next_batch(self, to_insert, custom_max_nodes=None):
        if custom_max_nodes is None:
            custom_max_nodes = self.max_batch_size_to_check


        values_to_bin = np.array([node.next_move_score for node in to_insert], dtype=np.float32)

        bin_indices = np.digitize(values_to_bin, self.bins)

        unique_bins, counts= np.unique(bin_indices, return_counts=True)

        num_to_check_for_termination = 0
        index_in_bins = 0
        bin_batch_became_full = -1
        for j in range(len(self.bin_arrays)):
            if bin_batch_became_full == -1:
                if index_in_bins != len(unique_bins) and unique_bins[index_in_bins] == j:
                    num_to_check_for_termination += counts[index_in_bins]
                    index_in_bins += 1

                if len(self.bin_arrays[j]) != 0:
                    num_to_check_for_termination += len(self.bin_arrays[j])

                if num_to_check_for_termination >= custom_max_nodes:
                    bin_batch_became_full = j
            else:
                if index_in_bins != len(unique_bins) and unique_bins[index_in_bins] == j:
                    self.bin_arrays[j] = np.append(self.bin_arrays[j], to_insert[bin_indices==j])
                    index_in_bins += 1

        if bin_batch_became_full == -1:
            bin_batch_became_full = len(self.bin_arrays)-1

        to_try_and_terminate = np.concatenate([self.bin_arrays[j] for j in range(bin_batch_became_full+1)] + [to_insert[bin_indices<=bin_batch_became_full]])

        list(map(terminate_if_possible, to_try_and_terminate))

        not_terminated_indices = np.array([False if node.terminated else True for node in to_try_and_terminate], dtype=np.bool)

        for j in range(bin_batch_became_full+1):
            if len(self.bin_arrays[j]) != 0:
                self.bin_arrays[j] = np.array([], dtype=np.object)

        return to_try_and_terminate[not_terminated_indices]