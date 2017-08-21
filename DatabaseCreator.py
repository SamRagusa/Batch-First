'''
Created on Aug 12, 2017

@author: SamRagusa
'''
import numpy as np
import chess
import chess.pgn
import random
from functools import reduce



def create_database_from_pgn(filename, filters=[], data_writer=None, switch_black_move_to_white=True, fen_elements_stored=[0,2],output_filenames=["board_config_database.csv"],error_output_filename="board_config_errors.csv",update_times=100):
    """
    @param filename The filename for the png file of chess games to read from.
    @param filters An array of functions which each require two parameters, one
    of a comma separated string of elements of of the games fen requested to be stored in the fen_elements_stored parameter.
    The other is the WinLossDrawData associated with that string.
    @param data_writer A function that takes in two parameters, the first of which is a dictionary mapping strings representations
    of the board (as determined by the other parameters) to objects of the WinLossDrawData class (internal).  The second parameter is the 
    output_filenames array.
    @param switch_black_move_to_white True if moves where it's blacks turn should be switched to a representation where it's whites turn.
    @param fen_elements_stored The elements to be stored (and used for duplicate detection) from the output of chess.Board().fen().spit(), which is of the format:
    r1bqkb1r/1p1n1pp1/p2ppn1p/8/3NP1P1/2N4Q/PPP1BP1P/R1B1K2R b KQkq - 1 9 2
    @param output_filenames An array of filenames (strings) to be passed to the data_writer parameter.
    @param update_times The number of times to print an updated percentage of games processed.
    @param error_output_filename The filename (string) of the file to write games that raised errors 
    
    TO DO:
    1) Add the ability to create additional entries in each output line based on given functions (for adding information about ELO or whatever else)
    2) Consider adding other methods (aside from switching from blacks move to whites) to convert boards to other known boards which are logically equivalent
    3) Confirm that the try statement actually catches and handles the errors, because it seems
    they're being handled internally within the chess package and the logs are being dumped to the Game objects .error array
    4) Round percentage of games processed to have two decimal places
    
    TESTS TO CREATE:
    1) Is switching blacks move to white working test
    2) Do filter functions get applied properly test
    3) Does it properly delete duplicates based on the fen elements stored
    
    """
    class WinLossDrawData:
        def __init__(self, outcome, reverse=False):
            self.wins = 0
            self.losses = 0
            self.draws = 0
            self.update(outcome, reverse)
        
        def __str__(self):
            return "{0},{1},{2}".format(self.wins, self.losses, self.draws)
        
        def update(self, outcome, reverse=False):
            if outcome == "1-0":
                if reverse:
                    self.losses = self.losses + 1
                else:
                    self.wins = self.wins + 1
            elif outcome == "0-1":
                if reverse:
                    self.wins = self.wins + 1
                else:
                    self.losses = self.losses + 1
            else:
                self.draws = self.draws + 1
    
    
    def line_counter(filename):
        """
        A function to count the number of lines in a file efficiently.
        """
        def blocks(files, size=65536):
            while True:
                b = files.read(size)
                if not b:
                    break
                yield b

        with open(filename, "r") as f:
            return sum(bl.count("\n") for bl in blocks(f))
    
    
    the_board = chess.Board()
    
    
    def black_fen_to_white():
        """
        Reformats and returns a the output from a given boards .fen() method from one where it's blacks turn,
        to one where it's whites turn.  This involves rotating the board along the x axis and switching 
        color of the pieces.
        """
        split_fen = the_board.fen().split()
#         if split_fen[1] == "w":
#             print("ERROR!  black_fen_to_white function recieving white players turn!")
#             print("Given fen:", fen)
        
        split_fen[0] = "/".join(list(reversed([row.swapcase() for row in split_fen[0].split("/")]))) 
        split_fen[1]  = "w"
        split_fen[2] = "".join(map(lambda x: x if (x in split_fen[2].swapcase()) else '', ['K','Q', 'k', 'q','-'])) #tested
        return ",".join(split_fen)
    
    
    
    pgn_file = open(filename)
    configs={}
    error_file = open(error_output_filename,'w')
    
    #set constants 
    counter = 0
    num_games_in_file = line_counter(filename)/16
    if num_games_in_file < update_times:
        update_times=num_games_in_file
        print("The frequency of updates chosen will produce an error,")
        print("so it has been changed to the number of games in the given file,")
    num_games_in_n_percent_of_file = num_games_in_file//update_times
    percent_increment = 100/update_times

    print("0.0% of games processed.")
    cur_game = chess.pgn.read_game(pgn_file)
    while not cur_game is None:
        try:
            the_result = cur_game.headers["Result"]
            for move in cur_game.main_line():
                the_board.push(move)
                if switch_black_move_to_white and not the_board.turn:
                    white_move_string = black_fen_to_white()
                else:
                    white_move_string = ",".join(the_board.fen().split())
    #                 if white_move_string.split(",")[1]=="b":
    #                     print("ERROR! Blacks move was not converted")
    #                     print("String: ", white_move_string)
    #                     print(cur_game.board().turn)
                    
                white_move_string = ",".join([info for index, info in enumerate(white_move_string.split(",")) if index in fen_elements_stored])
    
                reverse = True if (switch_black_move_to_white and not the_board.turn) else False
                if configs.get(white_move_string) is None:
                    configs[white_move_string]= WinLossDrawData(the_result, reverse)
                else:
                    configs[white_move_string].update(the_result, reverse)
#             if cur_game.errors != []:
#                 print(cur_game.errors)
        except:
            print("Error raised, game has been abandoned and written to error file.")
            error_file.write(list(cur_game.main_line()) + "\n")
        
        
        
        counter = counter + 1
        if counter % num_games_in_n_percent_of_file == 0:
            print(str(percent_increment*counter/num_games_in_n_percent_of_file) + "% of games processed.")
            
        #This is because it interprets the repeated result as it's own game
        #so it skips that game
        if chess.pgn.read_game(pgn_file) is None:
            break
        
        cur_game = chess.pgn.read_game(pgn_file)
        the_board.reset()

    print("Applying filters to data.")
    
    #Figure out a way of doing this with less of a space requirement
    to_delete = []
    if filters != []:
        for cur_str, data in configs.items():
            for filter in filters:
                if filter(cur_str, data):
                    to_delete.append(cur_str)
                    break
                
        for cur_str in to_delete:
            del configs[cur_str]

    
    print("Writing data to new file.")
    if data_writer is None:
        writer = open(output_filenames[0],'w')
        for cur_str, data in configs.items():
            writer.write(cur_str + "," + data.__str__() + "\n")  #data.__str__() should be done without the function call
        writer.close()
    else:
        data_writer(configs,output_filenames)
        
    
    pgn_file.close()
    error_file.close()





def indecisive_outcome_filter(str, data):
    """
    Filter out configurations where there is a tie between wins and losses.
    
    NOTES:
    1) Should think about adding more measures of indecisiveness
    """
    if data.losses == data.wins:
        return True
    return False


def opening_game_filter(str, data):
    """
    Returns True if the given string represents a game position which the AI will consider an "opening game position".
    """
    pass


def n_man_filter_creator(n):
    """
    Creates and returns a filter function for use in the create_database_from_pgn function.
    Returns True if the number of pieces in the given board configuration is greater than n,
    and False if there are less than or equal to n.
    """
    def filter(str, data):
        counter = 0
        for chr in str:
            if chr in "KQRBNPkqrbnp":
                counter = counter + 1
                if counter > n:
                    return False
        return True 
    return filter


def draw_game_filter(str, data):
    """
    Returns true if the the number of draws is greater than or equal to both
    the number of wins and the number of losses (in the data object).
    """
    if data.draws >= data.losses and data.draws >= data.wins:
        return True
    return False


def writer_creator(win_loss_data_to_str, map_entries_to_file_index, entry_str_manipulations=[]):
    """
    Creates a writer function for use in the create_database_from_pgn function. 
    """
    def the_writer(dict, filenames):
        writers = [open(filename,'w') for filename in filenames]
        entries = ["" for _ in range(len(dict))]
        for index, (str, data) in enumerate(dict.items()):
            for manipulation in entry_str_manipulations:
                str = manipulation(str)
            entries[index] = str + "," + win_loss_data_to_str(data)
            
        entry_dict = map_entries_to_file_index(entries)
        for str, index in entry_dict.items():
            if index != -1:
                writers[index].write(str + "\n")
            
        for writer in writers:
            writer.close()
            
    return the_writer


def str_ary_to_index_dict_function_creator(training_ratio=.65, validation_ratio=.25, testing_ratio=.1):
    """
    Creates and returns a function which takes in an array, and returns a dictionary 
    where the keys are the elements of the given array, and the values are the indices
    corresponding to which file to write the key to.
    """
    def str_ary_to_index_dict(ary):
        def decider(num):
            if num < testing_ratio:
                return 2
            elif num < testing_ratio + validation_ratio:
                return 1
            elif num < testing_ratio + validation_ratio + training_ratio:
                return 0
            else:
                return -1
        return {str : decider(rand) for str, rand in zip(ary,np.random.rand(len(ary)))}
    return str_ary_to_index_dict








manipulations = [(lambda y:(lambda x : x.replace(str(y), "1"*y)))(j) for j in range(2,9)]
manipulations.append(lambda x : x.replace("/", ""))

create_database_from_pgn("GMallboth.pgn",
    filters=[indecisive_outcome_filter, draw_game_filter, n_man_filter_creator(5)],
    fen_elements_stored=[0],
    data_writer=writer_creator(
        lambda x: "0" if x.wins > x.losses else "1",
        str_ary_to_index_dict_function_creator(),
        manipulations),
    output_filenames=["train_pipeline_eval.csv", "val_pipeline_eval.csv", "test_pipeline_eval.csv"])

