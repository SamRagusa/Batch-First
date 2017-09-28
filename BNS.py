'''
Created on Sep 1, 2017

@author: SamRagusa
'''

import chess
from negamax import Negamax
# import Human_Player
import random
import time
from functools import reduce
from chess.polyglot import zobrist_hash


class Player:
    """
    A class to be inherited by classes which represent a chess playing opponent.
    """

    def set_board(self, the_board):
        """
        Sets the Board object which is known by the AI.
        """
        self.board = the_board

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


class TranspositionTableEntry:
    """
    A class to store information about a node (board configuration) that has been
    previously computed.  It is used as the value in the transposition table.
    """

    def __init__(self, flag, value, depth):
        """
        INFO:
        1) The flag value is None if containing a value, True if containing
        an upperbound, and False if containing a lowerbound.
        """
        self.flag = flag
        self.depth = depth
        self.value = value


class BNS(Player):
    """
    A class representing a Chess playing AI implementing Negamax tree search with Alpha-Beta pruning.
    """

    def __init__(self, is_white, the_depth, board_heuristic=None, the_board=None, random_decrease=0, next_guess_heuristic=None):
        """
        Initialize the instance variables to be stored by the AI.
        """
        self._board = the_board
        self._depth = the_depth
        self._is_white = is_white
        if board_heuristic is None:
            self._heuristic = self.heuristic_creator(random_decrease=random_decrease)
        else:
            self._heuristic = board_heuristic

        # self.TEMP_TT_TABLE_USE_COUNTER =0

        self._tt = {}

        # not inf and -inf to try and fix the problem with try statement in make_move method
        WIN_VAL = 9999999
        LOSE_VAL = -9999999
        TIE_VAL = 0

        if self._is_white:
            self._outcome_map = {"1-0": WIN_VAL, "0-1": LOSE_VAL, "1/2-1/2": TIE_VAL}
        else:
            self._outcome_map = {"1-0": LOSE_VAL, "0-1": WIN_VAL, "1/2-1/2": TIE_VAL}

        if next_guess_heuristic is None:
            self._next_guess = lambda alpha,beta,count : alpha+(beta-alpha)*(count-1)/count
        else:
            self._next_guess = next_guess_heuristic

    def negamax(self, depth, alpha, beta, color):
        """
        A method implementing negamax tree search with alpha-beta pruning and
        transposition tables.

        NOTES:
        1) Maybe should be returning inf and -inf if not within the bounds
        """

        old_alpha = alpha

        board_hash = zobrist_hash(self._board)

        # Checks the transposition table for information on the current node
        tt_entry = self._tt.get(board_hash)
        if not tt_entry is None and tt_entry.depth >= depth:
            # self.TEMP_TT_TABLE_USE_COUNTER = self.TEMP_TT_TABLE_USE_COUNTER + 1
            if tt_entry.flag is None:
                return tt_entry.value
            elif tt_entry.flag:
                alpha = max(alpha, tt_entry.value)
            else:
                beta = min(beta, tt_entry.value)

            if alpha >= beta:
                return tt_entry.value

        if self._board.is_game_over():
            return color * self._outcome_map[self._board.result()]

        if depth == 0:
            return color * self._heuristic()

        best_value = float('-inf')
        for move in self._board.legal_moves:
            self._board.push(move)
            v = - self.negamax(depth - 1, -beta, -alpha, - color)
            best_value = max(best_value, v)
            alpha = max(alpha, v)
            self._board.pop()
            if alpha >= beta:
                break

        # Update the transposition table
        if best_value <= old_alpha:
            self._tt[board_hash] = TranspositionTableEntry(True, best_value, depth)
        elif best_value >= beta:
            self._tt[board_hash] = TranspositionTableEntry(False, best_value, depth)
        else:
            self._tt[board_hash] = TranspositionTableEntry(None, best_value, depth)

        return best_value



    def make_move(self, alpha=-4.4, beta=4.4):
        """
        NOTES:
        1) ***Pretty sure there is a serious bug somewhere within this method***
        1) This assumes that the evaluation function maps to (-4.4,4.4), which is for the temp evaluation function.
        This applies to most of the constants used in this method.
        """
        possible_moves = list(self._board.legal_moves)

        counter = 0
        while True:
            counter = counter + 1

            if counter % 100 == 0:
                print(counter)
                # print(alpha, beta, better_count)


            #Get the guess for the value to separate the nodes which are still being
            #searched through.
            if counter == 1:
                separator_value = self._heuristic()
            else:
                separator_value = self._next_guess(alpha, beta, len(possible_moves))

            moves_above = []
            for move in possible_moves:
                self._board.push(move)
                cur_search_value = -self.negamax(self._depth-1, -separator_value, .01-separator_value, -1)

                # print(separator_value, best_value, alpha, beta)

                self._board.pop()

                if cur_search_value >= separator_value:
                    moves_above.append(move)


            if len(moves_above) == 0:
                #Decrement beta (note that this is only a temporary method of doing this)
                beta = beta - .1

                if beta - alpha < .2:
                    return possible_moves[random.randint(0, len(possible_moves) - 1)], possible_moves

            else:
                alpha = separator_value
                possible_moves = moves_above

                if len(moves_above) == 1 or beta - alpha < .2:
                    return moves_above[random.randint(0, len(moves_above) - 1)], moves_above




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

            return score * (1 - random_decrease + random_decrease * random.random())

        return heuristic




board = chess.Board()

better_ai = BNS(True, 3, None, board)
other_better_ai = Negamax(True, 3, None, board, random_decrease=0)
worse_ai = Negamax(False, 2, None, board)
# human = Human_Player.Human_Player()
# human.set_board(board)

better_ai_wins = 0
worse_ai_wins = 0
move_counter = 0
last_time = None


NUM_GAMES = 10
PRINT_EVERY_MOVE = False
PRINT_AT_START_OF_GAME = True
PRINT_AT_MOVE_INCREMENT = False
STEP_PRINT_INCREMENT = 1
GAME_PRINT_INCREMENT = 1
MAX_MOVES = 250


for game_num in range(NUM_GAMES):
    if PRINT_AT_START_OF_GAME and game_num % GAME_PRINT_INCREMENT == 0:
        print("Starting game number", game_num)
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
        better_ai_move, possible_moves = better_ai.make_move()
        # print(board)
        other_better_ai_move = other_better_ai.make_move()
        # print(board)

        if not other_better_ai_move in possible_moves:
            print("BNS does not contain alpha-beta's move.")
            print(possible_moves)
            print(other_better_ai_move)
            print(board)
            print()
            board.push(other_better_ai_move)
            print(board)
            print()
        else:
            board.push(other_better_ai_move)

        if PRINT_EVERY_MOVE:
            print(board, "\n")

        if not board.is_game_over():
            if PRINT_AT_MOVE_INCREMENT and move_counter % STEP_PRINT_INCREMENT == 0:
                print("Making move number:", move_counter, ", average time per move: ",
                      (time.time() - last_time) / STEP_PRINT_INCREMENT)
                last_time = time.time()

            move_counter = move_counter + 1
            cur_game_move_counter = cur_game_move_counter + 1
            board.push(worse_ai.make_move())

            if PRINT_EVERY_MOVE:
                print(board, "\n")

    if board.result() == "1-0":
        better_ai_wins = better_ai_wins + 1
    elif board.result() == "0-1":
        worse_ai_wins = worse_ai_wins + 1

    board.reset()

print("Better player number of wins: ", better_ai_wins)
print("Worse player number of wins: ", worse_ai_wins)
print("Number of tie games ", NUM_GAMES - better_ai_wins - worse_ai_wins)
print()
print("Average moves per game:", move_counter / NUM_GAMES)
# print("Better AI used ", better_ai.TEMP_TT_TABLE_USE_COUNTER ,"TT lookups used")
# print("Worse AI used ", worse_ai.TEMP_TT_TABLE_USE_COUNTER,"TT lookups used")


