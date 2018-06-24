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
 

## Dependencies
The versions listed are known to work, but are not necessarily the only versions which will work.
- [TensorFlow](https://github.com/tensorflow/tensorflow) v1.8.0
- [NumPy](https://github.com/numpy/numpy) v1.14.3
- [Numba](https://github.com/numba/numba) v0.38.0
- [python-chess](https://github.com/niklasf/python-chess) v0.22.1
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