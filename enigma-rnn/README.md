Enigma-RNN: Decoding the Enigma with Recurrent Neural Networks
=======
See [blog post](https://greydanus.github.io/2017/01/07/enigma-rnn/)

About
--------
Contains the code for training an RNN (LSTM cell) to decode the Enigma cipher.

To download a trained model:
* Start in this directory
* **NOTE:** when upgrading this repo to work with TensorFlow 1.1 / Python 3.6, I was unable to load the models I had trained with old TensorFlow. I'm currently retraining the saved model. Until I have this fixed, you'll have to train your own models.
* `wget http://caligari.dartmouth.edu/~sgreydan/crypto-rnn/enigma-rnn/saved.tar.gz && tar -zxvf saved.tar.gz`

To test the model:
* Start in this directory
* `python main.py --train False`

To train the model:
* Start in this directory
* `python main.py`

Enigma cipher
--------
The [Enigma cipher](https://en.wikipedia.org/wiki/Enigma_machine) works like this.
![Enigma cipher](../static/enigma.gif?raw=true)

Dependencies
--------
* All code is written in Python 2.7 (for [crypto_enigma](https://pypi.python.org/pypi/crypto-enigma) to work) and TensorFlow 1.1. You will need:
 * NumPy
 * [TensorFlow](https://www.tensorflow.org/install/)
