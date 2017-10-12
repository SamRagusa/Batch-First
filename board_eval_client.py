'''
Created on Oct 5, 2017

@author: SamRagusa
'''

from tensorflow_serving.apis import regression_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations

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


    def score_board(self):
        """
        Scores the current representation of the board stored internally.  It does this by
        creating a Protocol Buffer of the board, sending it to the TensorFlow Serving instance,
        and returning the value received back.
        """
        example = self.request.input.example_list.examples.add()

        example.features.feature['str'].bytes_list.value.append(self.board.tf_representation())

        result = self.stub.Regress(self.request, 10.0)

        self.request.input.Clear()

        #Return the value, making sure that(for now) it is with respect to the initially specified player
        if (self.for_white and not self.board.turn) or (not self.for_white and self.board.turn):
            return - result.result.regressions[0].value
        else:
            return result.result.regressions[0].value

