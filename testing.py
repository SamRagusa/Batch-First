from numba_board import jitted_perft_test, create_board_state_from_fen, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS, BB_ALL, generate_legal_moves
import chess

#These fens come from the ChessProgramming Wiki, and can be found here: https://chessprogramming.wikispaces.com/Perft+Results
DEFAULT_TESTING_FENS = [
    # "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    # "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", #In check
    # "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", #Mirrored version of above
    # "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8 ",
    # "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
]


#These values come from the ChessProgramming Wiki, and can be found here: https://chessprogramming.wikispaces.com/Perft+Results
DEFAULT_FEN_PERFT_RESULTS = [
    # [20,400,8902,197281,4865609,119060324,3195901860,849989789566,2439530234167,693552859712417],
    # [48,2039,97862,4085603,193690690],
    # [14,191,2812,43238,674624,11030083, 178633661],
    [6,264,9467,422333,15833292,706045033],
    # [6,264,9467,422333,15833292,706045033],
    # [44,1486,62379,2103487,89941194],
    # [46,2079,89890,3894594,164075551,6923051137,287188994746,11923589843526,490154852788714],
]



def full_perft_tester(perft_function_to_test, fens_to_test=DEFAULT_TESTING_FENS,
                      perft_results=DEFAULT_FEN_PERFT_RESULTS, max_expected_boards_to_test=5000000):
    """
    Tests a given PERFT function by checking it's results against known results for several different boards and depths.
    :param perft_function_to_test: A function that takes 2 arguments, the first being a board's FEN representation
     as a string, and the second being the depth of the PERFT test to be done.
    :param fens_to_test: An iterable of strings, each a FEN representation of a board.
    :param perft_results: A list of the same length as fens_to_test, containing lists of integers, each corresponding
     to the expected PERFT result if tested at a depth of it's index
    :param max_expected_boards_to_test: The maximum expected PERFT result to test for a given board
    :return: True if all tests were passed, False if not
    """
    for cur_fen, cur_results in zip(fens_to_test, perft_results):
        for j, expected_result in enumerate(cur_results):
            if expected_result <= max_expected_boards_to_test:
                print(j)
                if perft_function_to_test(cur_fen, j+1) != expected_result:
                    print(perft_function_to_test(cur_fen, j + 1),expected_result)
                    return False
    return True

def alskdjflaksdjf():
    white_boards = []
    import random
    for i in range(100):
        board = chess.Board()
        for j in range(200):
            if j % 50 == 0:
                white_boards.append(board.copy())
            legal_move_list = list(board.legal_moves)
            if len(legal_move_list):
                board.push(random.choice(legal_move_list))
        # print(board)
        # print(board.fen())
    return white_boards

def compare_move_generations(boards):
    for cur_board in boards:
        board_fen = cur_board.fen()
        py_board = chess.Board(board_fen)
        my_board = create_board_state_from_fen(board_fen)
        my_len = len(list(generate_legal_moves(my_board, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS, BB_ALL, BB_ALL)))
        py_len = len(list(py_board.legal_moves))

        if my_len != py_len:
            print(py_board)
            print(py_board.fen, py_board.has_legal_en_passant())
            print(len(list(generate_legal_moves(my_board, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS, BB_ALL, BB_ALL))), len(list(py_board.legal_moves)))
            a = sorted([str(chess.Move(m.from_square, m.to_square, m.promotion)) for m in list(generate_legal_moves(my_board, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS, BB_ALL, BB_ALL))])
            b = sorted([str(f) for f in py_board.legal_moves])
            print(a)
            print(b)
            print(sorted(set(a) ^ set(b)))
            # print(sorted(set(b) - set(b)))
            print("\n"*5)

            # raise Exception('slkdjfalksd')

        ############################################REALLY SHOULD ALSO CHECK BB EQUIVILENCY

if __name__ == "__main__":
    ##############################################75 MOVE RULE!?!?!??!!!?!?!
    ##########REPITITION RULE


    # results = full_perft_tester(
    #         lambda fen, depth: jitted_perft_test
    #         (create_board_state_from_fen(fen), depth, OLD_BB_RANK_ATTACKS, OLD_BB_FILE_ATTACKS, OLD_BB_DIAG_ATTACKS))
    #
    # print(results)

    white_boards = alskdjflaksdjf()
    compare_move_generations(white_boards)
    print(white_boards)