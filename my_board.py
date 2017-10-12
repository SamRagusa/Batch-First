'''
Created on Oct 4, 2017

@author: SamRagusa
'''

import chess


def square_mirror(square):
    """Mirrors the square vertically."""
    return square ^ 0x38

SQUARES = [
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8] = range(64)
    
SQUARES_180 = [square_mirror(sq) for sq in SQUARES]

BB_SQUARES = [
    BB_A1, BB_B1, BB_C1, BB_D1, BB_E1, BB_F1, BB_G1, BB_H1,
    BB_A2, BB_B2, BB_C2, BB_D2, BB_E2, BB_F2, BB_G2, BB_H2,
    BB_A3, BB_B3, BB_C3, BB_D3, BB_E3, BB_F3, BB_G3, BB_H3,
    BB_A4, BB_B4, BB_C4, BB_D4, BB_E4, BB_F4, BB_G4, BB_H4,
    BB_A5, BB_B5, BB_C5, BB_D5, BB_E5, BB_F5, BB_G5, BB_H5,
    BB_A6, BB_B6, BB_C6, BB_D6, BB_E6, BB_F6, BB_G6, BB_H6,
    BB_A7, BB_B7, BB_C7, BB_D7, BB_E7, BB_F7, BB_G7, BB_H7,
    BB_A8, BB_B8, BB_C8, BB_D8, BB_E8, BB_F8, BB_G8, BB_H8
] = [1 << sq for sq in SQUARES]

BB_FILES = [
    BB_FILE_A,
    BB_FILE_B,
    BB_FILE_C,
    BB_FILE_D,
    BB_FILE_E,
    BB_FILE_F,
    BB_FILE_G,
    BB_FILE_H
] = [0x0101010101010101 << i for i in range(8)]



class MyBoard(chess.Board):

    def __str__(self):
        """
        Only written because I had issues printing the board while using Python 2.x
        """
        return super(MyBoard, self).__str__()


    def is_game_over(self, claim_draw=False):
        """
        The overriding of this method defined in chess.Board is temporary,
        and will not be done in the future.
        """
        # Seventyfive-move rule.
        # if self.is_seventyfive_moves():
        #     return True

        # Insufficient material.
        if self.is_insufficient_material():
            return True

        # Stalemate or checkmate.
        if not any(self.generate_legal_moves()):
            return True

        # # Draw claim.
        # if claim_draw and self.can_claim_draw():
        #     return True

        return False


    def tf_representation(self, promoted=False):
        """
        Returns the board in the format currently being accepted by the 
        neural network input function.  If it's blacks turn,
        the board is made to be as if it were whites turn.
        
        Current board format example:
        rnbqkbnrpppppppp11111111111111111111111111111111PPPPPPPPRNBQKBNR
        """
        
        #If it's whites turn
        if self.turn:
            return "".join(
                map(
                    lambda x: str(1) if not x else x.symbol(),
                    map(self.piece_at, SQUARES_180)))
        else:
            builder = []
            to_build = []
            for square in SQUARES_180:
                piece = self.piece_at(square)
    
                if not piece:
                    
                    to_build.append(str(1))
                else:
                    to_build.append(piece.symbol())
                    if promoted and BB_SQUARES[square] & self.promoted:
                        builder.append("~")
    
                if BB_SQUARES[square] & BB_FILE_H:
                    builder.append(to_build)
                    to_build = []
            
            return "".join(sum(reversed(builder),[]))


