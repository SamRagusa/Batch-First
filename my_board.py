'''
Created on Oct 4, 2017

@author: SamRagusa
'''

import chess



class MyBoard(chess.Board):

    def __str__(self):
        """
        Only written because I had issues printing the board while using Python 2.7
        """
        return super(MyBoard, self).__str__()


    def tf_representation(self, promoted=False):
        """
        Returns the board in the format currently being accepted by the
        neural network input function.  If it's blacks turn,
        the board is made to be as if it were whites turn.

        Current board format example:
        rnbqkbnrpppppppp11111111111111111111111111111111PPPPPPPPRNBQKBNR

        IMPORTANT NOTES:
        1) Converting boards to appear to always be from white's perspective shouldn't be done by creating
        an alternate string representation here, but should be transformed as described in the Trello card
        describing this problem.
        """

        #If it's whites turn
        if self.turn:
            return "".join(
                map(
                    lambda x: str(1) if not x else x.symbol(),
                    map(self.piece_at, chess.SQUARES_180)))
        else:
            builder = []
            to_build = []
            for square in chess.SQUARES_180:
                piece = self.piece_at(square)

                if not piece:

                    to_build.append(str(1))
                else:
                    to_build.append(piece.symbol())
                    if promoted and chess.BB_SQUARES[square] & self.promoted:
                        builder.append("~")

                if chess.BB_SQUARES[square] & chess.BB_FILE_H:
                    builder.append(to_build)
                    to_build = []

            return "".join(sum(reversed(builder),[]))


    def piece_capture_value_at(self, square, ):
        """
        A slightly modified version of the chess.Board.piece_type_at method.

        Notes:
        1) Should consider doing this with a map function to see if it's faster
        """

        mask = chess.BB_SQUARES[square]

        if not self.occupied & mask:
            return -.01
        elif self.pawns & mask:
            return 1
        elif self.knights & mask:
            return 3
        elif self.bishops & mask:
            return 3.1
        elif self.rooks & mask:
            return 5
        elif self.queens & mask:
            return 9
        elif self.kings & mask:
            return 12


    def score_move_based_on_captures(self, move):
        """
        IMPORTANT NOTES:
        1) This method will only be used temporarily, until the full implementation is ready.
        """
        piece_val_at_destination = self.piece_capture_value_at(move.to_square)

        # The -.01 is used because it is above any losing capture, but above any equal capture
        if piece_val_at_destination == -.01:
            return -.01

        return piece_val_at_destination - self.piece_capture_value_at(move.from_square)


    def ordered_move_generator_basic(self):
        """
        Creates a generator for legal moves based on captures

        IMPORTANT NOTES:
        1) This method will only be used temporarily, until the full implementation is ready.
        """
        move_dict = {move: self.score_move_based_on_captures(move) for move in self.generate_legal_moves()}

        #Loop the number of times equal to the number of unique values in move_dict
        while len(move_dict) != 0:
            max_val = float('-inf')
            max_val_moves = []

            for move, score in move_dict.iteritems():
                if score > max_val:
                    max_val = score
                    max_val_moves = [move]
                elif score == max_val:
                    max_val_moves.append(move)

            for move in max_val_moves:
                yield move
                del move_dict[move]


    def database_board_representation(self):
        """
        This method returns a string representation of the board for use during database generation.
        """
        builder = ["" for _ in range(64)]
        if self.ep_square is None:
            ep_square = -1
        else:
            ep_square = self.ep_square

        castling_rooks = []
        if self.castling_rights != 0:
            if self.castling_rights & chess.BB_A1:
                castling_rooks.append(chess.A1)
            if self.castling_rights & chess.BB_H1:
                castling_rooks.append(chess.H1)
            if self.castling_rights & chess.BB_A8:
                castling_rooks.append(chess.A8)
            if self.castling_rights & chess.BB_H8:
                castling_rooks.append(chess.H8)

        for square in chess.SQUARES_180:
            piece = self.piece_at(square)

            if not piece:
                if ep_square == square:
                    builder[square] = "1"
                else:
                    builder[square] = "0"

            else:
                if square in castling_rooks:
                    if piece.color == chess.WHITE:
                        builder[square] = 'C'
                    else:
                        builder[square] = 'c'
                else:
                    builder[square] = piece.symbol()

        return "".join(builder)

