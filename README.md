# SimpleANN

Simple single header file for neural network operations. Multiple threads should be able to propagate forward simultaneously. Remember to call outputWasRead() on ANNetwork to notify threads waiting access to the last layer.
