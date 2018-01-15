# Summary

In the Wu-Tang Clan song Da Mystery Of Chessboxin', you can find the following quote:

"The game of chess, is like a sword fight.  You must think first, before you move.  Toad style is immensely strong, and immune to nearly any weapon.  When it's properly used, it's almost invincible."

This engine will embody those ideas.

# Current Methods
- A convolutional neural network architecture inspired by the movement of chess pieces is used as the board evaluation function (detailed explanation coming very soon)
- Using Numba to compile all python code used for move generation to machine code with the LLVM compiler (complete negamax algorithm to come)
- Built a best-first negamax algorithm to better utilize batching for neural network inference and multiple CPU cores (partially Numba compiled)
- Using Best Node Search as the tree search algorithm
- Using TensorFlow Serving for neural network inference


# Things I'm Working On
- Utilizing a neural network to adjust separating values in the Best Node Search algorithm
- Designing and building a neural network based approach to move ordering (likely sharing first few convolutional layers with scoring network)

# Other Information
- Much of the code is still to come, but should be completed and/or uploaded before the end of January
- Special thanks to the python-chess package which most of my Numba move generation code is based on

Much, much more information and code to come!