import numpy as np
# import numba as nb

import chess

from batch_first.engine import ChessEngine, BatchFirstEngine
from batch_first.chestimator import get_inference_functions
from batch_first.anns.move_evaluation_ann import main as move_ann_main
# from batch_first.numba_board import vectorized_popcount




def play_one_game(engine1, engine2, print_info=False):
    """
    Given two objects which inherit the ChessEngine class this function will officiate one game of chess between the
    two engines.
    """

    the_board = chess.Board()

    halfmove_counter = 0
    engine1_turn = True
    while not the_board.is_game_over():
        if print_info:
            print(the_board)
            print("halfmove counter:", halfmove_counter)

        if engine1_turn:
            engine1.ready_engine()
            next_move = engine1.pick_move(the_board)
            engine1.release_resources()
        else:
            engine2.ready_engine()
            next_move = engine2.pick_move(the_board)
            engine2.release_resources()


        print("Making move:", next_move)
        the_board.push(next_move)

        engine1_turn = not engine1_turn
        halfmove_counter += 1

    if print_info:
        print("The game completed in %d halfmoves with result %s."%(halfmove_counter, the_board.result()))
        print(the_board)
    return the_board







class RandomEngine(ChessEngine):
    def pick_move(self, board):
        return np.random.choice(np.array([move for move in board.generate_legal_moves()]))


class StockFishEngine(ChessEngine):

    def __init__(self, stockfish_location, num_threads=1, move_time=15):

        self.stockfish_ai = chess.uci.popen_engine(stockfish_location)
        self.stockfish_ai.setoption({"threads" : num_threads})

        self.move_time = move_time


    def pick_move(self, board):
        self.stockfish_ai.position(board)
        return  self.stockfish_ai.go(movetime=self.move_time).bestmove





# temp_piece_values = np.array([900,500,300,300,100,500],dtype=np.float32)
# @nb.njit(nogil=True)
# def piece_sum_eval(pieces, occupied_bbs, unused):
#     piece_counts = vectorized_popcount(np.bitwise_and(pieces[...,1:], occupied_bbs))
#     return np.expand_dims(np.sum(temp_piece_values*(piece_counts[:,0].view(np.int8) - piece_counts[:,1].view(np.int8)).astype(np.float32),axis=1),axis=1)
#
# @nb.njit(nogil=True)
# def random_move_eval(unused_1, unused_2, unused_3, unused_4, num_moves_per_board):
#     return np.random.rand(np.sum(num_moves_per_board))





BOARD_EVAL_GRAPHDEF_FILE = "/srv/tmp/encoder_evaluation/conv_train_wide_and_deep_4/1526978123/tensorrt_eval_graph.pb"


BOARD_PREDICTOR, _, BOARD_PREDICTOR_CLOSER = get_inference_functions(BOARD_EVAL_GRAPHDEF_FILE, None)
MOVE_PREDICTOR, MOVE_PREDICTOR_CLOSER =  move_ann_main([True])



first_move_scoring_testing_filename = "/srv/databases/chess_engine/one_rand_per_board_data/move_scoring_testing_set_1.npy"

batch_first_engine = BatchFirstEngine(5, BOARD_PREDICTOR, MOVE_PREDICTOR, first_move_scoring_testing_filename)



# stockfish_engine = StockFishEngine("stockfish-8-linux/Linux/stockfish_8_x64")

random_engine = RandomEngine()


for j in range(1):
    play_one_game(batch_first_engine, random_engine, True)


BOARD_PREDICTOR_CLOSER()
MOVE_PREDICTOR_CLOSER()



