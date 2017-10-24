'''
Created on Sep 1, 2017

@author: SamRagusa
'''

import tensorflow as tf
import numpy as np

import chess
from chess.polyglot import zobrist_hash

from my_board import MyBoard
from negamax import Negamax
from board_eval_client import ANNEvaluator
from Random_AI import Random_AI
# import Human_Player

import random
import time
from functools import reduce



class Player:
    """
    A class to be inherited by classes which represent a chess playing opponent.
    """

    def set_board(self, the_board):
        """
        Sets the Board object which is known by the AI.
        """
        self._board = the_board

    def game_completed(self):
        """
        Should be overridden if AI implementing this class should be notified
        when a game ends, but before the board is wiped.
        """
        pass

    def make_move(self):
        """
        Makes a move on the Player object's chess.Board instance.
        """
        pass
    


class NewTranspositionTableEntry:
    """
    A class to store information about a node (board configuration) that has been
    previously computed.  It is used as the value in the transposition table.
    """

    def __init__(self, flag, depth, upper_bound=None, lower_bound=None):
        self.flag = flag
        self.depth = depth
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound



class BNS(Player):
    """
    A class representing a Chess playing AI implementing Negamax tree search with Alpha-Beta pruning.
    """

    def __init__(self, is_white, the_depth, board_heuristic, the_board=None, deepening=False , random_decrease=0):
        """
        Initialize the instance variables to be stored by the AI.
        """
        self._board = the_board
        self._depth = the_depth
        self._is_white = is_white
        self._deepening = deepening
        if board_heuristic is None:
            self._heuristic = self.heuristic_creator(random_decrease=random_decrease)
        else:
            self._heuristic = board_heuristic

        self._tt = {}

        # not inf and -inf to try and fix the problem with try statement in make_move method
        WIN_VAL = 9999
        LOSE_VAL = -9999
        TIE_VAL = 0

        if self._is_white:
            self._outcome_map = {"1-0": WIN_VAL, "0-1": LOSE_VAL, "1/2-1/2": TIE_VAL}
        else:
            self._outcome_map = {"1-0": LOSE_VAL, "0-1": WIN_VAL, "1/2-1/2": TIE_VAL}

        self._next_guess = lambda alpha,beta,count : alpha+(beta-alpha)*(count-1)/count


    def set_heuristic(self, new_heuristic):
        self._heuristic = new_heuristic


    def negamax(self, depth, alpha, beta, color):
        """
        A method implementing negamax tree search with alpha-beta pruning and
        transposition tables to decide what move to make given the current
        board configuration.

        NOTES:
        1) Maybe should be returning inf and -inf if not within the bounds

        TO-DO:
        1) have this take only one parameter instead of alpha and beta
        2) Have this run multithreaded
        """

        old_alpha = alpha

        board_hash = zobrist_hash(self._board)

        # Checks the transposition table for information on the current node
        tt_entry = self._tt.get(board_hash)
        if not tt_entry is None and tt_entry.depth >= depth:


            if not tt_entry.upper_bound is None:
                if tt_entry.upper_bound <= alpha:
                    return tt_entry.upper_bound
                beta = min(beta, tt_entry.upper_bound)

            if not tt_entry.lower_bound is None:
                if tt_entry.lower_bound >= beta:
                    return tt_entry.lower_bound
                alpha = max(alpha, tt_entry.lower_bound)


        if self._board.is_game_over():
            best_value =  color * self._outcome_map[self._board.result()]

        elif depth == 0:
            best_value = color * self._heuristic()
            # return self._heuristic()

        else:
            best_value = float('-inf')
            for move in self._board.legal_moves:
                self._board.push(move)
                v = - self.negamax(depth - 1, - beta, - alpha, - color)
                best_value = max(best_value, v)
                alpha = max(alpha, v)
                self._board.pop()
                if alpha >= beta:
                    break


        if best_value <= old_alpha:
            self._tt[board_hash] = NewTranspositionTableEntry(True, depth, best_value, None)
        elif best_value > old_alpha and best_value < beta:
            print("THIS SHOULD NEVER HAPPEN 2")
            self._tt[board_hash] = NewTranspositionTableEntry(None, depth, best_value, best_value)
        elif best_value >= beta:
            self._tt[board_hash] = NewTranspositionTableEntry(False, depth, None, best_value)
        else:
            print("THIS SHOULD NEVER HAPPEN 3")

        return best_value


    def best_node_search(self, depth, initial_separator, alpha=-3, beta=3):
        """
        TO-DO: 
        1) lookup and store TTs in this method
        """

        start_time = time.time()

        MIN_ALPHA_BETA_DIF = .00001#np.finfo(np.float32).eps  #np.nextafter(0, 1)

        possible_moves = list(self._board.legal_moves)

        counter = 0
        while True:
            counter = counter + 1

            if counter % 100 == 0:
                    print(counter, separator_value)
                    # print(alpha, beta, better_count)

            # Get the guess for the value to separate the nodes which are still being
            # searched through.
            if counter == 1:
                separator_value = initial_separator
            else:
                old_separator_value = separator_value
                separator_value = self._next_guess(alpha, beta, len(possible_moves))
                # if counter > 500:
                #     print("change in separator values:", old_separator_value - separator_value)
                #     print("beta - alpha:", beta-alpha)

                if abs(old_separator_value - separator_value) < MIN_ALPHA_BETA_DIF:

                    # negamax_values = []
                    # for move in possible_moves:
                    #     self._board.push(move)
                    #     negamax_values.append(-self.get_negamax_value(depth-1, -1))
                    #     self._board.pop()
                    # if len(set(negamax_values)) != 1:
                    #     print("FOUND ENDLESS LOOP 1")
                    #     print(alpha, beta, alpha - beta, separator_value, len(possible_moves))
                    #     print(set(negamax_values))

                    return possible_moves[random.randint(0, len(possible_moves) - 1)], possible_moves, separator_value

            moves_above = []
            for move in possible_moves:
                self._board.push(move)
                # print(separator_value)
                # print(MIN_ALPHA_BETA_DIF-separator_value)
                cur_search_value = -self.negamax(depth - 1, -separator_value, np.finfo(np.float32).eps - separator_value, -1)

                # print(separator_value, best_value, alpha, beta)

                self._board.pop()

                if cur_search_value >= separator_value:
                    moves_above.append(move)


            if len(moves_above) == 0:
#                 old_beta = beta
                beta = separator_value

                if beta - alpha <=  MIN_ALPHA_BETA_DIF:

                    # if len(possible_moves) > 1:
                    #     negamax_values = []
                    #     for move in possible_moves:
                    #         self._board.push(move)
                    #         negamax_values.append(-self.get_negamax_value(depth - 1, -1))
                    #         self._board.pop()
                    #     if len(set(negamax_values)) != 1:
                    #         print("Number of possible moves: 1", len(possible_moves))
                    #         print(old_beta,alpha, beta, alpha - beta, separator_value, len(possible_moves))
                    #         print(set(negamax_values))

                    return possible_moves[random.randint(0, len(possible_moves) - 1)], possible_moves, separator_value

            else:
#                 old_alpha = alpha
                alpha = separator_value
                possible_moves = moves_above

                if len(moves_above) == 1 or beta - alpha <= MIN_ALPHA_BETA_DIF:
                    # if len(moves_above) > 1:
                    #     negamax_values = []
                    #     for move in moves_above:
                    #         self._board.push(move)
                    #         negamax_values.append(-self.get_negamax_value(depth - 1, -1))
                    #         self._board.pop()
                    #     if len(set(negamax_values)) != 1:
                    #         print("Number of possible moves 2:", len(moves_above))
                    #         print(old_alpha, alpha, beta, alpha - beta, separator_value, len(possible_moves))
                    #         print(set(negamax_values))

                    return moves_above[random.randint(0, len(moves_above) - 1)], moves_above, separator_value


    def make_move(self, run_alpha=-12, run_beta=12):
        self._tt = {}

        cur_separator = self._heuristic()

        if self._deepening:
            for depth in range(1, self._depth):
                cur_separator = self.best_node_search(depth, cur_separator, alpha=run_alpha, beta=run_beta)[2]

        return self.best_node_search(self._depth, cur_separator, alpha=run_alpha, beta=run_beta)


    def full_negamax(self, depth, alpha, beta, color):
        board_hash = zobrist_hash(self._board)

        # Checks the transposition table for information on the current node
        tt_entry = self._tt.get(board_hash)
        if not tt_entry is None and tt_entry.depth >= depth:

            if tt_entry.flag:
                if tt_entry.value <= alpha:
                    return tt_entry.value
            elif not tt_entry.flag: #SHOULD BE ABLE TO BE JUST ELSE
                if tt_entry.value >= beta:
                    return tt_entry.value
            else:
                print("THIS SHOULD NEVER HAPPEN 1")


        if self._board.is_game_over():
            return color * self._outcome_map[self._board.result()]

        if depth == 0:
            return color * self._heuristic()

        best_value = float('-inf')
        for move in self._board.legal_moves:
            self._board.push(move)
            v = - self.full_negamax(depth - 1, - beta, - alpha, - color)
            # print("v =", v)
            best_value = max(best_value, v)
            alpha = max(alpha, v)
            self._board.pop()
            if alpha >= beta:
                break

        return best_value

    def get_negamax_value(self, depth, color):
        return self.full_negamax(depth, 10, -10, color)



    def heuristic_creator(self, piece_vals=[.1, .3, .45, .6, .9], random_decrease=.2):
        """
        Creates a basic heuristic if none was supplied.
        """

        def heuristic():
            score = reduce(
                lambda x, y: x + y,
                map(
                    lambda a, b, c: a * (b - c),
                    piece_vals,
                    map(
                        lambda d: len(self._board.pieces(d, self._is_white)),
                        [1, 2, 3, 4, 5]),
                    map(
                        lambda d: len(self._board.pieces(d, not self._is_white)),
                        [1, 2, 3, 4, 5])))

            return score * (1- random_decrease + random_decrease * random.random())

        return heuristic





def main(_):

    board = MyBoard()


    ann_evaluator_white = ANNEvaluator(board)
    ann_evaluator_black = ANNEvaluator(board, for_white=False)


    # better_ai = Negamax(True, 2, None, board, random_decrease=0.05)
    better_ai = BNS(True, 3, ann_evaluator_white.score_board, board, deepening=True)
    # better_ai = BNS(True, 4, None, board, random_decrease=0)#, deepening=True)
    # other_better_ai = BNS(True, 5, None, board, random_decrease=0)
    # other_better_ai = BNS(True, 8, ann_evaluator.get_win_probability_function(), board)
    # other_better_ai = Negamax(True, 3, ann_evaluator_white.test_get_win_probability_function, board)
    # worse_ai = BNS(False, 6, ann_evaluator_black.test_get_win_probability_function, board, random_decrease=.0)
    # worse_ai = BNS(False, 4, ann_evaluator_black.test_get_win_probability_function, board, random_decrease=.05)
    # worse_ai = Negamax(False, 2, None, board, random_decrease=.05)
    worse_ai = Random_AI()
    # human = Human_Player.Human_Player()
    # human.set_board(board)
    worse_ai.set_board(board)

    better_ai_wins = 0
    worse_ai_wins = 0
    move_counter = 0
    last_time = None
    better_ai_time = 0
    other_better_ai_time = 0

    """
    TO-DO:
    1) Test BNS without replacing TTs after every game and see if they still have no collisions
    """

    NUM_GAMES = 5
    PRINT_EVERY_MOVE = False
    PRINT_AT_END_OF_GAME = True
    PRINT_AT_MOVE_INCREMENT = False
    STEP_PRINT_INCREMENT = 1
    GAME_PRINT_INCREMENT = 1
    MAX_MOVES = 300


    for game_num in range(NUM_GAMES):
        if PRINT_AT_END_OF_GAME and game_num == 0:
            print("Starting game 0")
        cur_game_move_counter = 0
        while not board.is_game_over() and cur_game_move_counter < MAX_MOVES:
            if PRINT_AT_MOVE_INCREMENT and move_counter % STEP_PRINT_INCREMENT == 0:
                if move_counter == 0:
                    print("Making move number", move_counter)
                else:
                    print("Making move number:", move_counter, ", average time per move: ",
                          (time.time() - last_time) / STEP_PRINT_INCREMENT)
                last_time = time.time()

            move_counter = move_counter + 1
            cur_game_move_counter = cur_game_move_counter + 1
            # print(board)
            start_move_time = time.time()
            better_ai_move_to_make, better_ai_moves, _= better_ai.make_move()
            # better_ai_move_to_make = better_ai.make_move()
            move_time = time.time() - start_move_time
            # print("Deepening time:", move_time)
            better_ai_time = better_ai_time + move_time
            # print(board)

            # start_move_time = time.time()
            # other_better_ai_move = other_better_ai.make_move()
            # move_time = time.time() - start_move_time
            # print("Non-deepening time:", move_time)
            # other_better_ai_time =  other_better_ai_time + move_time
            # print(board)

            if False:#not other_better_ai_move in better_ai_moves:
                # print("BNS does not contain alpha-beta's move.")
                # print(better_ai_moves)
                # print(other_better_ai_move)
                # print(board)
                # print()
                # board.push(other_better_ai_move)
                # print(board)
                # print()
                pass
            else:
                board.push(better_ai_move_to_make)

            if PRINT_EVERY_MOVE:
                print(board)
                print()

            if not board.is_game_over():
                if PRINT_AT_MOVE_INCREMENT and move_counter % STEP_PRINT_INCREMENT == 0:
                    print("Making move number:", move_counter, ", average time per move: ",
                          (time.time() - last_time) / STEP_PRINT_INCREMENT)
                    last_time = time.time()

                move_counter = move_counter + 1
                cur_game_move_counter = cur_game_move_counter + 1
                board.push(worse_ai.make_move())

                if PRINT_EVERY_MOVE:
                    print(board)
                    print()

        winner=None
        if board.result() == "1-0":
            better_ai_wins = better_ai_wins + 1
            winner=True
        elif board.result() == "0-1":
            worse_ai_wins = worse_ai_wins + 1
            winner=False

        if PRINT_AT_END_OF_GAME and game_num % GAME_PRINT_INCREMENT == 0:
            if winner is None:
                winner = "tie game"
                print(board)
                # print(board.is_game_over())
                # print(board.result())
                # print(board.is_stalemate())
                # print(board.is_insufficient_material())
                # print(board.is_seventyfive_moves())
                # print(board.is_fivefold_repetition())
                # print(list(board.legal_moves))
            elif winner:
                winner = "better ai won"
                # print(board)
            else:
                winner = "worse ai won"
                # print(board)
            print("Finishing game number", game_num, winner)
        board.reset()

    print()
    print("Better player number of wins: ", better_ai_wins)
    print("Worse player number of wins: ", worse_ai_wins)
    print("Number of tie games ", NUM_GAMES - better_ai_wins - worse_ai_wins)
    print()
    print("Average moves per game:", move_counter / NUM_GAMES)
    print("Better player total move time:", better_ai_time)
    print("Other better player total move time:", other_better_ai_time)





if __name__ == '__main__':
    tf.app.run()