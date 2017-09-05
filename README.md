# ChessAI

This is a repository where I will be putting some code related to a chess engine I am building.

As of now, an inception based convolutional neural network will be used to score board configurations.

The current tree search implementation uses negamax with alpha-beta pruning.  This implementation was built for testing purposes, and will very likely not be used in the future.  

Currently the version of TensorFlow used in this code is 1.2.1.


# Current Plans

- Use inception based convolutional neural networks as a method of scoring a given chess board
- Design and build a neural network based approach to move ordering  (Currently thinking of using RNNs)
- Replace current method of tree searching with something optimized for use with neural networks
- Implement opening and closing game position tables

Much, much more information/code to come!
