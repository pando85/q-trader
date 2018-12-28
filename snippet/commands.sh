# Training with Q-learning
$ python tarin.py config/config.yaml

# Learning curve on training data
$ python learning_curve.py config/config.yaml ^GSPC

# Learning curve on test data
$ python learning_curve.py config/config.yaml ^GSPC_2011

# Plotting learning curve
$ python plot_learning_curve.py config/config.yaml ^GSPC

# Plotting trading on a model
$ python evaluate.py config/config2.yaml ^GSPC model_ep500

$ python plot_learning_curve.py config/config2.yaml ^GSPC

$ python plot_learning_curve.py config/config2.yaml ^GSPC_2011