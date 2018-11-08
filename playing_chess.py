import numpy as np

import chess
import chess.uci
import time

from batch_first.engine import ChessEngine, BatchFirstEngine
from batch_first.chestimator import get_inference_functions
from batch_first.anns.ann_creation_helper import combine_graphdefs, save_trt_graphdef, remap_inputs

from code_testing import weighted_piece_sum_creator


def play_one_game(engine1, engine2, print_info=False):
    """
    Given two objects which inherit the ChessEngine class this function will officiate one game of chess between the
    two engines.
    """
    the_board = chess.Board()
    prev_move_time = time.time()
    halfmove_counter = 0
    engine1_turn = True
    engine1.start_new_game()
    engine2.start_new_game()

    if print_info:
        print("%s\n%s\n"%(the_board, the_board.fen()))

    while not the_board.is_game_over():
        cur_engine = engine1 if engine1_turn else engine2

        cur_engine.ready_engine()
        next_move = cur_engine.pick_move(the_board)
        cur_engine.release_resources()

        if not the_board.is_legal(next_move):
            print("Exiting game due to player %d trying to push %s for the following board: \n%s"%(1+int(not engine1_turn), next_move, the_board))
            break

        the_board.push(next_move)
        engine1_turn = not engine1_turn
        halfmove_counter += 1

        if print_info:
            print("Player %d made move %s after %f time."%(1+int(not engine1_turn), str(next_move), time.time() - prev_move_time))
            print("Total halfmove count: %d\n%s\n%s\n"%(halfmove_counter, the_board, the_board.fen()))
            prev_move_time = time.time()

    if print_info:
        print("The game completed in %d halfmoves with result %s."%(halfmove_counter, the_board.result()))

    return the_board


class RandomEngine(ChessEngine):
    def pick_move(self, board):
        return np.random.choice(np.array(list(board.generate_legal_moves())))


class StockFishEngine(ChessEngine):

    def __init__(self, stockfish_location, num_threads=1, move_time=15, print_search_info=False):
        self.stockfish_ai = chess.uci.popen_engine(stockfish_location)
        self.stockfish_ai.setoption({"threads" : num_threads})

        self.info_handler = chess.uci.InfoHandler()
        self.stockfish_ai.info_handlers.append(self.info_handler)

        self.move_time = move_time

        self.print_info = print_search_info

    def pick_move(self, board):
        self.stockfish_ai.position(board)
        to_return = self.stockfish_ai.go(movetime=self.move_time).bestmove

        if self.print_info:
            print(self.info_handler.info)

        return to_return


def compete(player1, player2, pairs_of_games=1, print_results=True, print_games=False):
    outcomes = np.zeros([2,3],dtype=np.int32)

    result_indices = {"1-0": 0, "0-1": 1, "1/2-1/2": 2}
    match_info = [(0, player1, player2),
                  (1, player2, player1)]

    for _ in range(pairs_of_games):
        for i, p_white, p_black in match_info:
            outcomes[i, result_indices[play_one_game(p_white, p_black, print_games).result()]] += 1

    if print_results:
        for j in range(2):
            print("When player %d was white there was: %d wins, %d losses, and %d ties"%(j+1, outcomes[j, 0], outcomes[j, 1], outcomes[j, 2]))

    return outcomes


def pseudo_random_move_eval(*args):
    """
    NOTES:
    1) This function uses np.linspace instead of randomly generated values to maintain the deterministic behavior of the
    tree search (only helpful when tracking down a bug).
    """
    return lambda x: np.linspace(0, 1, np.sum(x[1]))


if __name__ == "__main__":
    MAX_SEARCH_BATCH_SIZE = 1024


    GRAPHDEF_FILENAMES = [
        "/srv/tmp/encoder_evaluation_helper/only_one_mate_weight_decrement_1.13/1540312680",
        "/srv/tmp/move_scoring_helper_current/new_data_one_pass_14_no_final_bn_scaling/new_data_one_pass_14_no_final_bn_scaling.15/1541546710",
    ]
    OUTPUT_MODEL_PATH = "/srv/tmp/combining_graphs_1"
    OUTPUT_MODEL_FILENAME = "COMBINED_OUTPUT_TEST_1.pbtxt"

    OUTPUT_NODE_NAMES = ["Squeeze", "requested_move_scores"]

    PREFIXES = ["value_network", "policy_network"]

    # combine_graphdefs(
    #     GRAPHDEF_FILENAMES,
    #     OUTPUT_MODEL_PATH,
    #     OUTPUT_MODEL_FILENAME,
    #     OUTPUT_NODE_NAMES,
    #     name_prefixes=PREFIXES,
    # )

    # remap_inputs(OUTPUT_MODEL_PATH + "/" +  OUTPUT_MODEL_FILENAME, OUTPUT_MODEL_PATH, "REMAPPED_TEST_1.pbtxt")

    OUTPUT_NODE_NAMES = ["%s/%s"%(prefix,name) for name, prefix in zip(OUTPUT_NODE_NAMES, PREFIXES)]
    TRT_OUTPUT_FILENAME = "COMBINED_TRT_TEST_1.pbtxt"
    # save_trt_graphdef(OUTPUT_MODEL_PATH + "/" +  OUTPUT_MODEL_FILENAME,
    #                OUTPUT_MODEL_PATH,
    #                TRT_OUTPUT_FILENAME,
    #                OUTPUT_NODE_NAMES,
    #                trt_memory_fraction=.4,
    #                max_batch_size=int(1.25*MAX_SEARCH_BATCH_SIZE))


    MOVE_SCORING_TEST_FILENAME = "/srv/databases/lichess/lichess_db_standard_rated_2018-07_first_100k_games.npy"
    ZERO_VALUE_BOARD_FILENAME = "/srv/databases/has_zero_valued_board/combined_zero_boards.npy"


    BOARD_PREDICTOR, MOVE_PREDICTOR, PREDICTOR_CLOSER = get_inference_functions(OUTPUT_MODEL_PATH + "/" + TRT_OUTPUT_FILENAME, session_gpu_memory=.65)


    search_depth = 6
    batch_first_engine = BatchFirstEngine(
        search_depth,
        # weighted_piece_sum_creator()[0],
        # pseudo_random_move_eval,
        BOARD_PREDICTOR,
        MOVE_PREDICTOR,
        bin_database_file="deeper_network_1.npy",
        max_batch_size=MAX_SEARCH_BATCH_SIZE,
        saved_zero_shift_file="new_zero_board_thing_1.npy"
    )

    stockfish_engine = StockFishEngine("stockfish-8-linux/Linux/stockfish_8_x64", move_time=1, print_search_info=True)

    competition_results = compete(batch_first_engine, stockfish_engine, print_games=True)

    PREDICTOR_CLOSER()

