'''
Created on Sep 1, 2017

@author: SamRagusa
'''

import chess    
# import Human_Player
# from Random_AI import Random_AI
import random

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
        self._heuristic = board_heuristic
    
    
        #not inf and -inf to try and fix the problem involving try statement in make_move method
        WIN_VAL = 9999999
        LOSE_VAL = -9999999
        TIE_VAL = 0
        
        if self._is_white:
            self._outcome_map = {"1-0" : WIN_VAL, "0-1" : LOSE_VAL, "1/2-1/2" : TIE_VAL}
        else:
            self._outcome_map = {"1-0" : LOSE_VAL, "0-1" : WIN_VAL, "1/2-1/2" : TIE_VAL}


    def negamax(self, depth, alpha, beta, color):
        """
        A method implementing negamax with alpha-beta pruning to decide what move to make given 
        the current board configuration. 
        
        NOTES:
        1) Implement Transposition Tables
        """
        if self._board.is_game_over():
            return color * self._outcome_map[self._board.result()]

        if depth == 0:
            return color * self._heuristic(self._board)
        
        
        best_value = float('-inf')
        for move in self._board.legal_moves:
            self._board.push(move)
            v = - self.negamax(depth-1,-beta, -alpha, - color)
            best_value = max(best_value, v)
            alpha = max(alpha, v)
            self._board.pop()
            
            if alpha >= beta:
                break
        return best_value
        

    def make_move(self):
        """
        TO-DO:
        1) Figure out the cause of the try statement requirement and how to eliminate it
        """
        move_to_make = None
        alpha = float('-inf')
        v = float('-inf')
        
        for move in self._board.legal_moves:
            self._board.push(move)
            negamax_results = -self.negamax(self._depth-1, float('-inf'), -alpha, -1)
            if negamax_results > v:
                v = negamax_results
                move_to_make = move
            alpha = max(alpha, v)
            self._board.pop()
        try:
            self._board.push(move_to_make)
        except:
            temp_legal_move_list = self._board.legal_moves
#             print(move_to_make)
#             print(self._board)
#             print(self._board.is_game_over())
#             print(len(temp_legal_move_list))
#             print("is white:", self._is_white)

            #Makes a random move because this error shouldn't occur in the long term
            self._board.push(list(temp_legal_move_list)[random.randint(0,len(temp_legal_move_list)-1)])
        


def test_heuristic_generator(is_white, piece_vals=[.1, .3,.45,.6,.9], random_decrease=.2):
    """
    Generates and returns a basic heuristic for board evaluation.
    """
    def test_heuristic(board):
        score = 0
        for index, val in enumerate(piece_vals):
            score = score + val * (len(board.pieces(index + 1,is_white)) - len(board.pieces(index + 1,not is_white)))
        return score * (1 - random_decrease + random_decrease*random.random())
    return test_heuristic
    
    
board =chess.Board()
 
better_ai = Negamax(True, 3, test_heuristic_generator(True), board)
worse_ai = Negamax(False, 2, test_heuristic_generator(False), board)
# rand = Random_AI()
# human = Human_Player.Human_Player()
# human.set_board(board)
# rand.set_board(board)
 
better_ai_wins = 0
worse_ai_wins = 0
NUM_GAMES = 6
move_counter = 0
PRINT_EVERY_MOVE = False
STEP_PRINT_INCREMENT = 100
GAME_PRINT_INCREMENT = 1
for game_num in range(NUM_GAMES):
    if game_num % GAME_PRINT_INCREMENT ==0:
        print("Starting game number", game_num)
    while not board.is_game_over():
        if move_counter % STEP_PRINT_INCREMENT == 0:
            print("Making move number", move_counter)
        
        move_counter = move_counter + 1
        better_ai.make_move()
        
        if PRINT_EVERY_MOVE:
            print(board, "\n")
     
        if not board.is_game_over():
            if move_counter % STEP_PRINT_INCREMENT == 0:
                print("Making move number", move_counter)
            
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



