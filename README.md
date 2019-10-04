# rf_pool
Perform pooling across receptive fields of different sizes and locations in 
convolutional neural networks (CNNs).

This package allows users to perform various pooling operations within receptive
fields tiling the output of a convolutional layer. Whereas typical max-pooling 
operations occur in blocks of pixels (e.g., 2x2 MaxPool), rf-pooling occurs 
within individual receptive fields covering variable areas of the image space.
Various initializations for the receptive field array are available including
uniform, foveated, and tiled. Furthermore, rf_pool supports building models 
with feedforward or Restricted Boltzmann Machine (RBM) layers among a variety
of other options.
