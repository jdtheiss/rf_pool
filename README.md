# rf_pooling
Perform pooling across receptive fields of different sizes in detection layer

This package allows users to perform probabilistic max-pooling (Lee et al., 2008),
stochastic max-pooling (Zeiler & Fergus, 2013), divisive normalization (Heeger, 1992),
or average pooling across receptive fields tiling the detection layer. The make_RFs
function allows users to build receptive fields of various sizes in an organization
similar to that observed in cortex (small receptive fields centrally with larger
receptive fields peripherally). The rf_pool function then performs one of the
above types of pooling across the receptive fields. Downsampling is performed by
considering blocks of (n,n) units in the detection layer connecting to units in
the pooling layer. A receptive field that covers m blocks in the detection
layer will then have m units in the pooling layer.
