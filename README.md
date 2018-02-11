# Batch First
In the Wu-Tang Clan song [Da Mystery Of Chessboxin'](https://youtu.be/pJk0p-98Xzc "YouTube Link"), you can find the following quote:

"The game of chess, is like a sword fight.  You must think first, before you move.  Toad style is immensely strong, and immune to nearly any weapon.  When it's properly used, it's almost invincible."

The Batch First engine will embody these ideas.

# Current Methods
- Using a **convolutional neural network architecture inspired by the movement of chess pieces** for both board evaluation and move scoring
- Implementing a best-first negamax algorithm to utilize batching for neural network inference on GPU, and to better utilize multiple CPU cores.  I'm calling it **Batch First Search** 
- Using Numba to compile all python code used for move generation to machine code with the LLVM compiler
- Using TensorFlow Serving for neural network inference

# Things I'm Working On
- Converting the array oriented parts of the negamax algorithm to implementations which use multiple CPU cores and SIMD instructions (mainly through Numba's Vectorize)
- JIT compiling (with Numba) the entirety of the negamax algorithm implementation
- The tree search algorithm which will call the zero-window negamax search (e.g. MTD(f) or Best Node Search)
- A formal description of the engine.  This will either be a full paper, or a detailed explanation of the core components (e.g. a proof of correctness for the Batch First Search algorithm, and reasoning behind the neural network architectures used)

# Dependencies
The versions listed are the versions I'm currently using, but are not necessarily the only versions which will work.
- [TensorFlow](https://github.com/tensorflow/tensorflow) v1.4.1 (moving to v1.5 soon)
- [NumPy](https://github.com/numpy/numpy) v1.13.3
- [Numba](https://github.com/numba/numba) v0.35.0
- [python-chess](https://github.com/niklasf/python-chess) v0.22.1
- [khash_numba](https://github.com/synapticarbors/khash_numba)
- [TensorFlow Serving](https://github.com/tensorflow/serving) v0.4.0 (No longer used during runtime, and will be phased out completely soon)

#Notes For Awesome Developers/People
This engine is still in development, and thus needs work.  If you have any ideas, questions, are interested in potentially working on the engine, or anything else, I encourage you to send me an email!  
   

# Other Information
- Trained neural networks will be uploaded when hyperparameter tuning has been completely implemented/completed
- Special thanks to the python-chess package which most of my move generation code is currently based on




