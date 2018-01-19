'''
Created on Oct 5, 2017

@author: SamRagusa
'''

import numpy as np
from tensorflow_serving.apis import regression_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations
from numba_board import get_info_for_tensorflow

import chess
from my_board import MyBoard

class ANNEvaluator:
    def __init__(self, the_board, host="172.17.0.2", port=9000, model_name='win_loss_ann', model_desired_signature='serving_default', for_white=True):
        self.host = host
        self.port = port
        self.model_name = model_name
        self.model_desired_signature=model_desired_signature
        self.channel = implementations.insecure_channel(host, port)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)
        self.request = regression_pb2.RegressionRequest()
        self.request.model_spec.name = self.model_name
        self.request.model_spec.signature_name = self.model_desired_signature
        self.board = the_board
        self.for_white = for_white


    def for_numba_score_batch(self, game_nodes, timeout=10):
        """
        Scores an array of GameNode JitClass objects using TensorFlow Serving.  It does this by creating a
        Protocol Buffer of the boards, and sending it to the TensorFlow Serving instance,
        then returning the value received back.

        IMPORTANT NOTES:
        1) The current implementation seems like a very slow method of doing this, and will (very) likely not be
        used long term (it was created to verify other implementations before going too in depth).
        Using fancy Numpy and TensorFlow Serving techniques (likely buffers), I will hopefully send all this
        information at once without any seemingly tedious method calls like get_info_for_tensorflow(board_state).
        The functionality of this method (really) should eventually be running from a compiled function/method,
        which seems like cannot be done currently using just the core functionality of Numba,
        and will likely have to be written in C from a Numba compiled function.
        """

        self.request.input.Clear()

        for node in game_nodes:
            board_state = node.board_state
            example = self.request.input.example_list.examples.add()

            test = get_info_for_tensorflow(board_state)

            example.features.feature["occupied_w"].int64_list.value.append(test[0])
            example.features.feature["occupied_b"].int64_list.value.append(test[1])
            example.features.feature["kings"].int64_list.value.append(test[2])
            example.features.feature["queens"].int64_list.value.append(test[3])
            example.features.feature["rooks"].int64_list.value.append(test[4])
            example.features.feature["bishops"].int64_list.value.append(test[5])
            example.features.feature["knights"].int64_list.value.append(test[6])
            example.features.feature["pawns"].int64_list.value.append(test[7])
            example.features.feature["castling_rights"].int64_list.value.append(test[8])
            example.features.feature["ep_square"].int64_list.value.append(test[9])

        result = self.stub.Regress(self.request, timeout)
        return result.result.regressions


    def score_board(self, timeout=10):
        """
        Scores the internally stored MyBoard object.  It does this by creating a Protocol Buffer of the board,
        sending it to the TensorFlow Serving instance, then returning the value received back.

        NOTES:
        1) This implementation is slow and is no longer in use other than maintaining old code and as a simple
        workaround for some speed-insensitive code.
        """
        example = self.request.input.example_list.examples.add()
        # not confident in my choice of encoding, but it seems to fix the issue...
        example.features.feature['str'].bytes_list.value.append(bytes(self.board.tf_representation(), encoding="utf-8"))
        result = self.stub.Regress(self.request, timeout)

        self.request.input.Clear()

        #Return the value, making sure that(for now) it is with respect to the initially specified player
        if (self.for_white and not self.board.turn) or (not self.for_white and self.board.turn):
            return - result.result.regressions[0].value
        else:
            return result.result.regressions[0].value

