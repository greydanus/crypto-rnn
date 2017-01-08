Autokey-NN: Decoding the Autokey cipher with Recurrent Neural Networks
=======
See [blog post](https://greydanus.github.io/2017/01/01/enigma-rnn/)

About
--------
Contains the code for training an RNN (LSTM cell) to decode the Autokey cipher.

To test the model:
* Start in this directory
* `python main --train False`

To train the model:
* Start in this directory
* `python main`

The [Autokey cipher](https://en.wikipedia.org/wiki/Autokey_cipher) works like the Vigenere cipher except that instead of reapeating the keyphrase, it appends the plaintext to the end of keyphrase.

Dependencies
--------
* All code is written in Python 2.7 and TensorFlow. You will need:
 * Numpy
 * [TensorFlow](https://www.tensorflow.org/versions/master/get_started/os_setup.html#pip_install)
