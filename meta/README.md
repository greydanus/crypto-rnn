Note
=======

Training stats get saved here. When you train a model, it will write loss to `loss_hist.txt` and accuracies to `acc_hist.txt`. It will also save the amount of time required to generate data, time required to do a forward/backward pass over the model, and total time of a single train step.

The Jupyter notebooks in the `figures` directory of this repo use these text files to generate loss and accuracy plots. These files are also useful for checking progress when you're running on a GPU cluster and can't see your job directly.