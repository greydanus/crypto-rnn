Enigma-NN: Decoding the Enigma with Recurrent Neural Networks
=======
See [blog post](https://greydanus.github.io/2017/01/01/enigma-rnn/)

About
--------
Contains the code for training an RNN (LSTM cell) to decode the Enigma cipher.

**To download a pretrained model.**
* Start in this directory
* `mkdir models`
* `cd models`
* `wget http://caligari.dartmouth.edu/~sgreydan/crypto-nn/enigma-rnn/models/checkpoint`

To test the model:
* Start in this directory
* `python main --train False`

To train the model:
* Start in this directory
* `python main`

The [Enigma cipher](https://en.wikipedia.org/wiki/Enigma_machine) works like this.
![Enigma cipher](static/enigma.gif?raw=true)

Dependencies
--------
* All code is written in Python 2.7 and TensorFlow. You will need:
 * Numpy
 * [TensorFlow](https://www.tensorflow.org/versions/master/get_started/os_setup.html#pip_install)
