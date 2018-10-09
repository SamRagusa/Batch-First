# Batch First
Batch First is a JIT compiled chess engine which traverses the search tree in batches in a best-first manner, allowing for neural network batching, asynchronous GPU use, and vectorized CPU computations.  Utilizing NumPy's powerful ndarray for representing boards, TensorFlow for neural networks computations, and Numba to JIT compile it all, Batch First balances the rapid prototyping desired in artificial intelligence with the runtime efficiency needed for a competitive chess engine.


### Engine Characteristics
The following table highlights a few key aspects of the Batch First engine. 

*Characteristic* | *Explanation / Reasoning* 
:---: | ---
**Written in Python** | Python is both easily readable and extremely flexible, but it's runtime speed has historically prevented it's use in competitive chess engines.  Through the use of high level packages, Batch First balances runtime speed and code readability
**JIT Compiled** | To avoid the execution time of Python, Numba is used to JIT compile the Python code to native machine instructions using the LLVM compiler infrastructure
**Batched ANN Inference** | By using a `K-best-first search` algorithm, evaluation of boards and moves can be done in batches, both minimizing the effect of latency and maximizing the throughput for GPUs
**Priority Bins** | Best-first algorithms such as SSS* are often disregarded due to the cost of maintaining a global list of open nodes in priority order.  This is addressed by instead using a pseudo-priority order, separating nodes based on their binned heuristic values
**Vectorized Asynchronous CPU Operations** | Through a combination of NumPy and Numba, the array oriented computations are vectorized and compiled to run while the ANNs are being evaluated, and with the Python GIL released


## Board Evaluation CNN

### Input Features/Training Data
Boards are given to the ANN as an 8x8x15 one-hot encoding, which consists of 12 feature planes for each piece and color,
2 for each player's rooks with the ability to castle, and 1 for en passant capture squares.  The label for each board
is the precomputed value of the board by an established engine (currently StockFish is used).    


### Input Layers
To model the movement of chess pieces, a novel architecture is used where the ANNs 'first layer' is
replaced with 9 convolutional layers.  When concatenated, the squares considered by the input convolutions
centered at any given square are the squares which could contain a piece able to threaten that square,
and the square itself.

This is accomplished by a set of dilated padded convolutional layers, and can be explained in two parts.
- An n-dilated convolutional layer with kernel size 3x3 will consider all potential rank, file,
and diagonal threats n squares away from the kernel's center.  Having 7 of these
with dilation factors of 1-7 encompasses all potential movement of every piece but the knight.  

- To capture the movement of the knight, two convolutional layers with kernel size 2x2 and dilation factor
2x4 and 4x2 are used.  Combined, the kernels of these two layers consider only 
the squares a knight has the potential to move to.  

The following diagram shows the structure of the input layers:
 
|                   |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |
|:-----------------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|  **Kernel Size**  | 3x3 | 3x3 | 3x3 | 3x3 | 3x3 | 3x3 | 3x3 | 2x2 | 2x2 |
|**Dilation Factor**| 1x1 | 2x2 | 3x3 | 4x4 | 5x5 | 6x6 | 7x7 | 2x4 | 4x2 |


### Loss
The network learns to score boards through classification methods, this is accomplished by learning 
an ordinal representation of the training boards.  More specifically, it learns to classify pairs of boards as having
the first or second board be preferable.  

Extending this to batches of data, the calculated value of each board is compared against every other board in the batch
(with unequal desired score).

Thus for a training batch B with desired and computed values D and C (shaped \[1,n\]), the network learns by
minimizing the following:

<!--- 
LowerTriangular(D-D^T\neq0)*CrossEntropy(S(C-C^T),D-D^T>0)
-->
![equation](https://latex.codecogs.com/gif.latex?LowerTriangular%28D-D%5ET%5Cneq0%29*CrossEntropy%28S%28C-C%5ET%29%2CD-D%5ET%3E0%29)

Where S is the sigmoid function, CrossEntropy is a function who's first and second parameters are logits and labels
respectively, LowerTriangular is a function which replaces entries above the main diagonal with zeros, 
multiplication is done element-wise (Hadamard product), and subtraction is calculated using broadcasting
(similar to NumPy's broadcasting).

If the values of D are unique, batch B will produce n(n-1)/2 pairs of boards to compare.
  
  
## Dependencies
The versions listed are known to work, but are not necessarily the only versions which will work.
- [TensorFlow](https://github.com/tensorflow/tensorflow) v1.10.0
- [NumPy](https://github.com/numpy/numpy) v1.14.3
- [Numba](https://github.com/numba/numba) v0.39.0
- [SciPy](https://github.com/scipy/scipy) v1.1.0
- [python-chess](https://github.com/niklasf/python-chess) v0.20.1
- [khash_numba](https://github.com/synapticarbors/khash_numba)

The tools listed below can be used, but are not needed.  They provide speed improvements to the engine.
- [TensorRT](https://developers.googleblog.com/2018/03/tensorrt-integration-with-tensorflow.html) to optimize TensorFlow graphs for inference
- [Intel Python Distribution](https://software.intel.com/en-us/distribution-for-python) for speed improvements to NumPy and Numba


## Miscellaneous Information
- If you use this engine or the ideas it's built around to build or experiment with something interesting, please be vocal about it!
- If you have any questions, ideas, or just want to say hi, feel free to get in contact with me!
- Trained neural networks are not yet included (they still have some room to improve), but can/will be added if requested
- Special thanks to the [python-chess](https://github.com/niklasf/python-chess) package which is heavily relied upon throughout the engine, and which most of the move generation code is based on


## License
Batch First is licensed under the GPL 3.  The full text can be found in the LICENSE.txt file.