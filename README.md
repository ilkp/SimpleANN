# SimpleANN

Simple single header file for neural network operations. Multiple threads should be able to propagate forward simultaneuosly. Remember to call outputWasRead() on ANNetwork to release to notify threads waiting access to the last layer.
