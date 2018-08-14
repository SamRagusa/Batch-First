import numpy as np

import chess
import chess.uci
import time


from batch_first.engine import ChessEngine, BatchFirstEngine
from batch_first.chestimator import get_inference_functions
from batch_first.anns.ann_creation_helper import save_model_as_graphdef_for_serving



def play_one_game(engine1, engine2, print_info=False):
    """
    Given two objects which inherit the ChessEngine class this function will officiate one game of chess between the
    two engines.
    """
    the_board = chess.Board()
    prev_move_time = time.time()
    halfmove_counter = 0
    engine1_turn = True

    if print_info:
        print("%s\n%s\n"%(the_board, the_board.fen()))

    while not the_board.is_game_over():
        cur_engine = engine1 if engine1_turn else engine2

        cur_engine.ready_engine()
        next_move = cur_engine.pick_move(the_board)
        cur_engine.release_resources()

        the_board.push(next_move)
        engine1_turn = not engine1_turn
        halfmove_counter += 1

        if print_info:
            player_str = "Player 1" if not engine1_turn else "Player 2"
            print("%s made move %s after %f time." % (player_str, str(next_move), time.time() - prev_move_time))
            print("Total halfmove count: %d"%halfmove_counter)
            print("%s\n%s\n" % (the_board, the_board.fen()))
            prev_move_time = time.time()

    if print_info:
        print("The game completed in %d halfmoves with result %s."%(halfmove_counter, the_board.result()))

    return the_board





class RandomEngine(ChessEngine):
    def pick_move(self, board):
        return np.random.choice(np.array(list(board.generate_legal_moves())))



class StockFishEngine(ChessEngine):

    def __init__(self, stockfish_location, num_threads=1, move_time=15):

        self.stockfish_ai = chess.uci.popen_engine(stockfish_location)
        self.stockfish_ai.setoption({"threads" : num_threads})

        self.move_time = move_time


    def pick_move(self, board):
        self.stockfish_ai.position(board)
        return  self.stockfish_ai.go(movetime=self.move_time).bestmove



if __name__ == "__main__":
    # board_eval_model_path = "/srv/tmp/encoder_evaluation_helper/diff_fixed_filter_6.4/1534156862"
    # save_model_as_graphdef_for_serving(
    #     model_path=board_eval_model_path,
    #     output_model_path=board_eval_model_path,
    #     output_filename="tensorrt_eval_graph.pb",
    #     output_node_name="logit_layer/MatMul",
    #     max_batch_size=5000)
    #
    # move_scoring_model_path = "/srv/tmp/move_scoring_helper/to_from_square_3/1532363846"
    # save_model_as_graphdef_for_serving(
    #     model_path=move_scoring_model_path,
    #     output_model_path=move_scoring_model_path,
    #     output_filename="tensorrt_move_scoring_graph.pb",
    #     output_node_name="GatherNd_1",
    #     trt_memory_fraction=.35,
    #     max_batch_size=7500)

    BOARD_EVAL_GRAPHDEF_FILENAME = "/srv/tmp/encoder_evaluation_helper/diff_fixed_filter_6.4/1534156862/tensorrt_eval_graph.pb"
    MOVE_SCORING_GRAPHDEF_FILENAME = "/srv/tmp/move_scoring_helper/to_from_square_3/1532363846/tensorrt_move_scoring_graph.pb"


    BOARD_PREDICTOR, MOVE_PREDICTOR, PREDICTOR_CLOSER = get_inference_functions(BOARD_EVAL_GRAPHDEF_FILENAME, MOVE_SCORING_GRAPHDEF_FILENAME)

    first_move_scoring_testing_filename = "/srv/databases/chess_engine/one_rand_per_board_data/move_scoring_testing_set_1.npy"


    batch_first_engine = BatchFirstEngine(5,
                                          BOARD_PREDICTOR,
                                          MOVE_PREDICTOR,
                                          bin_database_file=first_move_scoring_testing_filename,
                                          bin_output_filename="NEXT_BINS")

    stockfish_engine = StockFishEngine("stockfish-8-linux/Linux/stockfish_8_x64")
    random_engine = RandomEngine()



    play_one_game(batch_first_engine, stockfish_engine, True)



    PREDICTOR_CLOSER()
