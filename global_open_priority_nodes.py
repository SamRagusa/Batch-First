import numpy as np
from numba import njit
from concurrent.futures import ThreadPoolExecutor, wait
import bisect

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

def terminate_if_possible(game_node):
    if game_node.terminated:
        return True

    if not game_node.parent is None:
        if terminate_if_possible(game_node.parent):
            game_node.terminated = True
            game_node.parent.children_left -= 1
            return True
    return False




def find_not_terminating_indices(node_array):
    return np.array([should_not_terminate(node) for node in node_array] ,np.bool)



def split_by_bins(to_insert, bin_indices):
    """
    @:return An array of tuples, the first element of which is a bin index, and the second is a numpy array of
    nodes which belong to the bin specified by the first element
    """
    sorted_indices = bin_indices.argsort()
    sorted_bin_indices = bin_indices[sorted_indices]
    cut_indices = np.flatnonzero(sorted_bin_indices[1:] != sorted_bin_indices[:-1])+1
    slice_things = np.r_[0, cut_indices, len(sorted_bin_indices)+1]
    out = [(sorted_bin_indices[i], to_insert[sorted_indices[i:j]]) for i,j in zip(slice_things[:-1], slice_things[1:])]
    return out




class PriorityBins:
    def __init__(self, bins, max_batch_size_to_check, num_workers_to_use, search_extra_ratio, testing=False):
        self.bins = bins

        #This should very likely be a 2d numpy array and not start with size 0, as it would allow for faster
        self.bin_arrays = [np.array([], dtype=np.object) for _ in range(len(bins)+1)]
        self.max_batch_size_to_check = max_batch_size_to_check  #MAYBE RENAME THIS TO MIN NOT MAX

        self.num_workers_to_use = num_workers_to_use

        #This is sorted
        self.non_empty = [] #This needs to be a NumPy array
        self.max_prospective_nodes_open = int(search_extra_ratio * max_batch_size_to_check)
        self.testing = testing


    def __len__(self):
        total_len = 0
        for bin_index in self.non_empty:
            total_len += len(self.bin_arrays[bin_index])
        return total_len

    def is_empty(self):
        if self.non_empty == []:
            return True
        return False


    def largest_bin(self):
        if len(self.non_empty) == 0:
            return 0
        return max([len(self.bin_arrays[index]) for index in self.non_empty])


    # @profile
    def insert_batch_and_get_next_batch(self, to_insert, custom_max_nodes=None):
        """
        NOTES:
        1) This method can easily be sped up, but I don't believe that when everything else is optimized, this
        can be done with a singly threaded method (such as this) and not bottleneck everything else.
        2) Since it checks for termination only once, often it returns batches much smaller than desired,
        which results in a much slower negamax search

        """
        if custom_max_nodes is None:
            custom_max_nodes = self.max_prospective_nodes_open


        values_to_bin = np.array([node.next_move_score for node in to_insert], dtype=np.float32)

        bin_indices = np.digitize(values_to_bin, self.bins)

        if len(to_insert) != 0:
            bins_to_insert = split_by_bins(to_insert, bin_indices)
        else:
            bins_to_insert = []

        num_to_check_for_termination = 0
        index_in_bins = 0
        bin_batch_became_full = -1
        for j in range(len(self.bin_arrays)):
            if bin_batch_became_full == -1:
                if index_in_bins != len(bins_to_insert) and bins_to_insert[index_in_bins][0] == j:
                    num_to_check_for_termination += len(bins_to_insert[index_in_bins][1])
                    index_in_bins += 1

                if len(self.bin_arrays[j]) != 0:
                    num_to_check_for_termination += len(self.bin_arrays[j])

                if num_to_check_for_termination >= custom_max_nodes:
                    bin_batch_became_full = j
            else:
                if index_in_bins != len(bins_to_insert) and bins_to_insert[index_in_bins][0] == j:
                    if not j in self.non_empty:
                        bisect.insort(self.non_empty, j) #When this method is used this is only used for is_empty method
                    self.bin_arrays[j] = np.append(self.bin_arrays[j], bins_to_insert[index_in_bins][1])
                    index_in_bins += 1

        if bin_batch_became_full == -1:
            bin_batch_became_full = len(self.bin_arrays)-1

        to_try_and_terminate = np.concatenate([self.bin_arrays[j] for j in range(bin_batch_became_full+1)] + [bin[1] for bin in bins_to_insert if bin[0] <=bin_batch_became_full])

        not_terminated_indices = np.array(list(map(should_not_terminate, to_try_and_terminate)), dtype=np.bool)

        for j in range(bin_batch_became_full+1):
            if len(self.bin_arrays[j]) != 0:
                self.non_empty.remove(j)#When this method is used this is only used for is_empty method
                self.bin_arrays[j] = np.array([], dtype=np.object)

        return to_try_and_terminate[not_terminated_indices]



    def best_bin_iterator(self, bins_to_insert):
        """
        Iterate through the bins being given to insert, and the non-empty bins stored in priority order.  If there are
        non-empty stored bins where nodes given to insert belong, the nodes being given for insertion will be given
        before the bin stored.

        @:return A size two tuple, the first element is a bool saying if it's in the bins to insert or in the
        bins stored, the second element being the index in whichever array of bins it's referring.

        NOTES:
        1) If there are no nodes to insert this will not work
        """
        index_in_non_empty_bins = 0
        for index_in_sorted_bins, to_insert in enumerate(bins_to_insert):
            while index_in_non_empty_bins < len(self.non_empty) and self.non_empty[index_in_non_empty_bins] < to_insert[0]:
                yield (True, self.non_empty[index_in_non_empty_bins])
                index_in_non_empty_bins += 1
            yield (False, index_in_sorted_bins)


    # @profile
    def new_insert_batch_and_get_next_batch(self, to_insert):
        """
        TO DO FOR SPEED IMPROVEMENTS:
        1) Don't create threads for every bin, instead wait until a specified number of nodes are acquired,
        then start a thread.  Right now it's spending most of it's time creating threads.
        1) When a thread is done checking for termination and has a specified number of elements which were not
        terminated, the computation (up to giving nodes to GPU) for the next iteration should be started for the
        non-terminating nodes.  Not sure yet how this can be done most efficiently, it might make sense to wait until
        the rest of the core components have been optimized to decide how to proceed, since its
        implementation could be tailored to work on the more time consuming operations.  But if you have a good idea
        of how to do this don't let that stop you!
        2) Figure out if it's possible for there to be a node which should be terminated, who's values stored could
        update already terminated parent nodes, providing more information than they were initially terminated with.

        IMPORTANT NOTES:
        1) This currently is not fully working, it loses nodes (I'm pretty sure), and sometimes results in an
        incorrect (and often impossible) result from the negamax search

        """
        values_to_bin = np.array([node.next_move_score for node in to_insert], dtype=np.float32)

        bin_indices = np.digitize(values_to_bin, self.bins)

        bins_to_insert = split_by_bins(to_insert, bin_indices)

        next_batch = np.array([], np.object)

        array_iterator = self.best_bin_iterator(bins_to_insert)

        with ThreadPoolExecutor(max_workers=self.num_workers_to_use) as executor:
            stored_indices_used = []
            to_insert_indices_not_used = list(range(len(bins_to_insert))) #This should be a NumPy array
            num_acquired = 0
            prospective_nodes_acquired = 0
            future_to_node_id = {}
            num_task_starts = 0


            try:
                while prospective_nodes_acquired < self.max_prospective_nodes_open:
                    node_array_id = next(array_iterator)
                    num_task_starts += 1
                    if node_array_id[0]:
                        NEXT_BEST_NODE_SET = self.bin_arrays[node_array_id[1]]
                    else:
                        NEXT_BEST_NODE_SET = bins_to_insert[node_array_id[1]][1]

                    prospective_nodes_acquired += len(NEXT_BEST_NODE_SET)
                    the_future = executor.submit(find_not_terminating_indices, NEXT_BEST_NODE_SET)

                    #This should never happen, but I'm not super farmiliar with the concurrent.futures package so I'm putting it here for now
                    if self.testing and not future_to_node_id.get(the_future) is None:
                        print("future attempting to be up in dictionary already existed")
                    future_to_node_id[the_future] = node_array_id




                while len(future_to_node_id) != 0:
                    done_futures, _ = wait(future_to_node_id, return_when="FIRST_COMPLETED")

                    for cur_future in done_futures:
                        is_stored, index_in_array = future_to_node_id[cur_future]

                        if is_stored:
                            done_nodes = self.bin_arrays[index_in_array]
                            stored_indices_used.append(index_in_array)
                        else:
                            done_nodes = bins_to_insert[index_in_array][1]
                            to_insert_indices_not_used.remove(index_in_array)

                        done_not_terminating_mask = cur_future.result()

                        #I think this will be faster than concatenation done once at the end, though not sure yet
                        next_batch = np.append(next_batch, done_nodes[done_not_terminating_mask])
                        cur_num_not_terminating = np.sum(done_not_terminating_mask)

                        prospective_nodes_acquired -= len(done_not_terminating_mask) - cur_num_not_terminating
                        num_acquired += cur_num_not_terminating


                        del future_to_node_id[cur_future]

                    if num_acquired >= self.max_batch_size_to_check:
                        break

                    while prospective_nodes_acquired < self.max_prospective_nodes_open:
                        num_task_starts += 1
                        node_array_id = next(array_iterator)
                        if node_array_id[0]:
                            NEXT_BEST_NODE_SET = self.bin_arrays[node_array_id[1]]
                        else:
                            NEXT_BEST_NODE_SET = bins_to_insert[node_array_id[1]][1]

                        prospective_nodes_acquired += len(NEXT_BEST_NODE_SET)
                        the_future = executor.submit(find_not_terminating_indices, NEXT_BEST_NODE_SET)

                        # This should never happen, but I'm not super familiar with the concurrent.futures package so I'm putting it here for now
                        if self.testing and not future_to_node_id.get(the_future) is None:
                            print("future attempting to be up in dictionary already existed")
                        future_to_node_id[the_future] = node_array_id
            except StopIteration:
                self.non_empty = []
                stored_indices_used = []

                done_futures, _ = wait(future_to_node_id, return_when="ALL_COMPLETED")

                for cur_future in done_futures:
                    is_stored, index_in_array =  future_to_node_id[cur_future]

                    if is_stored:
                        done_nodes = self.bin_arrays[index_in_array]
                        stored_indices_used.append(index_in_array)
                    else:
                        done_nodes = bins_to_insert[index_in_array][1]
                        to_insert_indices_not_used.remove(index_in_array)

                    next_batch = np.append(next_batch, done_nodes[cur_future.result()])
            else:
                list(map(lambda x : x.cancel(), future_to_node_id)) #NEED TO MAKE SURE I DON'T NEED TO HANDLE THE CANCELLATION WITHIN THE find_not_terminating_indices FUNCTION

            # finally:
                for index in stored_indices_used:
                    self.non_empty.remove(index)


                for index in to_insert_indices_not_used:
                    if bins_to_insert[index][0] in self.non_empty:
                        self.bin_arrays[bins_to_insert[index][0]] = np.append(
                            self.bin_arrays[bins_to_insert[index][0]],bins_to_insert[index][1])
                    else:
                        self.bin_arrays[bins_to_insert[index][0]] = bins_to_insert[index][1]
                        bisect.insort(self.non_empty, bins_to_insert[index][0])

        array_iterator.close()

        return next_batch

