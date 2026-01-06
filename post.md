Autoencoder based signal reconstructor
Train an autoencoder network to reconstruct a signal from an altered signal. Then, lock the encoder and train additional layers to perform a downstream task. In this case, we train a linear regression model to predict the high power frequency component of the signal.

Data generation:
Initially we generate a single channel signal with a dominant non-stationary frequency component and a secondary non-stationary frequency component. This is done multiple times to create a dataset of signals. each of these source signal is split into a batch of 30 seconds. Each batch is altered a set number of times using paramatized alterations. The alterations include:
- Random noise injection
- Phase shift
- Frequency shift
- Amplitude shift

Each altered signal is paired with its source signal to create a dataset of (altered_signal, source_signal) pairs.
These pairs are then used to train the autoencoder network.