Autokey-RNN: Decoding the Autokey cipher with Recurrent Neural Networks
=======
See [blog post](https://greydanus.github.io/2017/01/07/enigma-rnn/)

About
--------
Contains the code for training an RNN (LSTM cell) to decode the Autokey cipher.

To download a trained model:
* Start in this directory
* **NOTE:** when upgrading this repo to work with TensorFlow 1.1 / Python 3.6, I was unable to load the models I had trained with old TensorFlow. I'm currently retraining the saved model. Until I have this fixed, you'll have to train your own models.
* `wget http://caligari.dartmouth.edu/~sgreydan/crypto-rnn/autokey-rnn/saved.tar.gz && tar -zxvf saved.tar.gz`

To test the model:
* Start in this directory
* `python main.py --train False`

To train the model:
* Start in this directory
* `python main.py`

The [Autokey cipher](https://en.wikipedia.org/wiki/Autokey_cipher) works like the Vigenere cipher except that instead of reapeating the keyphrase, it appends the plaintext to the end of keyphrase.

Dependencies
--------
* All code is written in Python 3.6 and TensorFlow 1.1. You will need:
 * NumPy
 * [TensorFlow](https://www.tensorflow.org/install/)
