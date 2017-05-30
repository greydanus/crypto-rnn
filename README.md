Crypto-RNN: Decoding Polyalphabetic Ciphers with Recurrent Neural Networks
=======
See [blog post](https://greydanus.github.io/2017/01/07/enigma-rnn/)

About
--------
This repo contains three (very similar) implementations of an LSTM-based deep learning model for decoding polyalphabetic ciphers. The first two, [vigenere-rnn](https://github.com/greydanus/crypto-rnn/tree/master/vigenere-rnn) and [autokey-rnn](https://github.com/greydanus/crypto-rnn/tree/master/autokey-rnn) are light proof-of-concept models. The third, [enigma-rnn](https://github.com/greydanus/crypto-rnn/tree/master/enigma-rnn) is much larger and more complex. It needs to be, because decoding the Enigma cipher is a very complex process.

Vigenere cipher
--------
The [Vigenere cipher](https://en.wikipedia.org/wiki/Vigen%C3%A8re_cipher) works like this (where we're encrypting plaintext "CALCUL" with keyword "MATHS" (repeated)). The [Autokey cipher](https://en.wikipedia.org/wiki/Autokey_cipher) is a slightly more secure variant.
![Vigenere cipher](static/vigenere.gif?raw=true)

Enigma cipher
--------
The [Enigma cipher](https://en.wikipedia.org/wiki/Enigma_machine) works like this.
![Enigma cipher](static/enigma.gif?raw=true)

Dependencies
--------
* All code is written in Python 3.6 and TensorFlow 1.1. You will need:
 * NumPy
 * [TensorFlow](https://www.tensorflow.org/install/)
