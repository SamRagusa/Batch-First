'''
Created on Sep 1, 2017

@author: SamRagusa
'''

import chess
# import Human_Player
# from Random_AI import Random_AI
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




class Negamax(Player):
    """
    A class representing a Chess playing AI implementing Negamax tree search with Alpha-Beta pruning.   
    """
    
    
    
    def __init__(self, is_white, the_depth, board_heuristic, the_board=None):
        """
        Initialize the instance variables to be stored by the AI. 
        """
        self._board = the_board
        self._depth = the_depth
        self._is_white = is_white
        if board_heuristic is None:
            self._heuristic = self.heuristic_creator()
        else:
            self._heuristic = board_heuristic
#         self._heuristic = board_heuristic
        
#         self.TEMP_TT_TABLE_USE_COUNTER =0
        
        self._tt = {}
    
    
        #not inf and -inf to try and fix the problem with try statement in make_move method
        WIN_VAL = 9999999
        LOSE_VAL = -9999999
        TIE_VAL = 0
        
        if self._is_white:
            self._outcome_map = {"1-0" : WIN_VAL, "0-1" : LOSE_VAL, "1/2-1/2" : TIE_VAL}
        else:
            self._outcome_map = {"1-0" : LOSE_VAL, "0-1" : WIN_VAL, "1/2-1/2" : TIE_VAL}

    def negamax(self, depth, alpha, beta, color):
        """
        A method implementing negamax tree search with alpha-beta pruning and
        transposition tables to decide what move to make given the current
        board configuration. 
        """   
        
        old_alpha = alpha
        
        cur_board_fen = self._board.board_fen()
        
        #Checks the transposition table for information on the current node
        tt_entry = self._tt.get(cur_board_fen)
        if not tt_entry is None and tt_entry.depth >= depth:
#             self.TEMP_TT_TABLE_USE_COUNTER = self.TEMP_TT_TABLE_USE_COUNTER + 1
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
            v = - self.negamax(depth-1,-beta, -alpha, - color)
            best_value = max(best_value, v)
            alpha = max(alpha, v)
            self._board.pop()
            if alpha >= beta:
                break
            

        #Update the transposition table
        if best_value <= old_alpha:
            self._tt[cur_board_fen] = TranspositionTableEntry(True, best_value, depth)
        elif best_value >= beta:
            self._tt[cur_board_fen] = TranspositionTableEntry(False, best_value, depth)
        else:
            self._tt[cur_board_fen] = TranspositionTableEntry(None, best_value, depth)
        
        
        return best_value
        

    def make_move(self):
        """
        TO-DO:
        1) **IMPORTANT** Eliminate try statement 
        2) Add current node to TT after computations
        """
        move_to_make = None
        alpha = float('-inf')
#         beta = float('inf')
        v = float('-inf')
        
#         moves = self._board.legal_moves
#         num_moves_prev = len(self._board.legal_moves)
#         move_counter = 0
        for move in self._board.legal_moves:
#             move_counter = move_counter +1
            self._board.push(move)
            negamax_results = -self.negamax(self._depth-1, float('-inf'), -alpha, -1)
            if negamax_results > v:
                v = negamax_results
                move_to_make = move
            alpha = max(alpha, v)
            self._board.pop()
        try:
#             if num_moves_prev != move_counter:
#                 print("Error!  Move list is being altered!")
            self._board.push(move_to_make)
        except:
            temp_legal_move_list = self._board.legal_moves
#             if num_moves_prev != move_counter:
#                 print("Error!  Move list is being altered!")
#             print(move_to_make)
#             print(self._board)
#             print(self._board.is_game_over())
#             print("Number of moves:", len(self._board.legal_moves))
#             print("is white:", self._is_white)

            #Makes a random move because this error shouldn't occur in the long term
            self._board.push(list(temp_legal_move_list)[random.randint(0,len(temp_legal_move_list)-1)])
            
    def heuristic_creator(self, piece_vals=[.1, .3,.45,.6,.9], random_decrease=.2):
        """
        Creates a basic heuristic if none was supplied.
        """
        def heuristic():
            score = reduce(
                lambda x,y: x+y,
                map(
                    lambda a,b,c: a*(b-c),
                    piece_vals,
                    map(
                        lambda d: len(self._board.pieces(d, self._is_white)),
                        [1,2,3,4,5]),
                    map(
                        lambda d: len(self._board.pieces(d, not self._is_white)),
                        [1,2,3,4,5])))
            
            return score * (1 - random_decrease + random_decrease*random.random())
        
        return heuristic
        


 

 
board =chess.Board()
 
better_ai = Negamax(True, 3, None, board)
worse_ai = Negamax(False, 2, None, board)
# rand = Random_AI()
# human = Human_Player.Human_Player()
# human.set_board(board)
# rand.set_board(board)
 
better_ai_wins = 0
worse_ai_wins = 0
NUM_GAMES = 200
move_counter = 0
PRINT_EVERY_MOVE = False
PRINT_AT_START_OF_GAME = True
PRINT_AT_MOVE_INCREMENT = False
STEP_PRINT_INCREMENT = 500
GAME_PRINT_INCREMENT = 10
last_time = None
for game_num in range(NUM_GAMES):
    if PRINT_AT_START_OF_GAME and game_num % GAME_PRINT_INCREMENT ==0:
        print("Starting game number", game_num)
    while not board.is_game_over():
        if PRINT_AT_MOVE_INCREMENT and move_counter % STEP_PRINT_INCREMENT == 0:
            if move_counter == 0:
                print("Making move number", move_counter)
            else:
                print("Making move number:", move_counter, ", average time per move: ", (time.time()-last_time)/STEP_PRINT_INCREMENT)
            last_time = time.time()
        
        move_counter = move_counter + 1
        better_ai.make_move()
        
        if PRINT_EVERY_MOVE:
            print(board, "\n")
     
        if not board.is_game_over():
            if PRINT_AT_MOVE_INCREMENT and move_counter % STEP_PRINT_INCREMENT == 0:
                print("Making move number:", move_counter, ", average time per move: ", (time.time()-last_time)/STEP_PRINT_INCREMENT)
                last_time = time.time()
            
            move_counter = move_counter + 1
            worse_ai.make_move()
            
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
print("Average moves per game:", move_counter/NUM_GAMES)
# print("Better AI used ", better_ai.TEMP_TT_TABLE_USE_COUNTER ,"TT lookups used")
# print("Worse AI used ", worse_ai.TEMP_TT_TABLE_USE_COUNTER,"TT lookups used")


