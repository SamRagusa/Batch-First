# Batch First
In the Wu-Tang Clan song [Da Mystery Of Chessboxin'](https://youtu.be/pJk0p-98Xzc "YouTube Link"), you can find the following quote:

> The game of chess, is like a sword fight.  You must think first, before you move.  Toad style is immensely strong, and immune to nearly any weapon.  When it's properly used, it's almost invincible.

The Batch First engine will embody these ideas.

# Current Methods
- Using a **convolutional neural network architecture inspired by the movement of chess pieces** for both board evaluation and move scoring
- Implementing a **zero-window k-best-first negamax search** algorithm to utilize batching for neural network inference run asynchronously on GPU, as well as better utilize multiple CPU cores
- Using Numba to compile all Python code used for move generation to machine code with the LLVM compiler
- Using a framework similar to MTD(f) to converge towards a boards negamax value

# Things Being Working On
- Converting the array oriented parts of the negamax algorithm to implementations which better utilize SIMD instructions (mainly through Numba's Vectorize)
- JIT compiling (with Numba) the entirety of the negamax algorithm implementation
- A semi-formal explanation of the engine's core components (e.g. a simple description of the negamax algorithm, and reasoning behind the neural network architecture used)

# Dependencies
The versions listed are the versions I'm currently using, but are not necessarily the only versions which will work.
- [TensorFlow](https://github.com/tensorflow/tensorflow) v1.7.0
- [NumPy](https://github.com/numpy/numpy) v1.14.2
- [Numba](https://github.com/numba/numba) v0.36.2
- [python-chess](https://github.com/niklasf/python-chess) v0.22.1
- [khash_numba](https://github.com/synapticarbors/khash_numba)

I run this using the [Intel Distribution For Python](https://software.intel.com/en-us/distribution-for-python) for the speed improvements to NumPy and Numba.   

# Notes For Awesome Developers/People
This engine is still in development, and thus needs work.  If you have any ideas, questions, are interested in potentially working on the engine, or anything else, I encourage you to send me an email!  
   

# Other Information
- Trained neural networks will be uploaded when hyperparameter tuning has been completely implemented/completed
- Special thanks to the python-chess package which most of my move generation code is currently based on




