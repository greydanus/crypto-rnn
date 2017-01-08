Vigenere-NN: Decoding the Vigenere cipher with Recurrent Neural Networks
=======
See [blog post](https://greydanus.github.io/2017/01/07/enigma-rnn/)

About
--------
Contains the code for training an RNN (LSTM cell) to decode the Vigenere cipher.

To test the model:
* Start in this directory
* `python main --train False`

To train the model:
* Start in this directory
* `python main`

The [Vigenere cipher](https://en.wikipedia.org/wiki/Vigen%C3%A8re_cipher) works like this.
![Vigenere cipher](static/vigenere.gif?raw=true)

Dependencies
--------
* All code is written in Python 2.7 and TensorFlow. You will need:
 * Numpy
 * [TensorFlow](https://www.tensorflow.org/versions/master/get_started/os_setup.html#pip_install)
