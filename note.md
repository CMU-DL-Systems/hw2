# Init parameters
There are four ways to initialize parameters: xavier_uniform, xavier_normal, kaiming_uniform and kaiming_normal.  
Return tensors. Device and dtype are determined.

# Implement nn modules
Implement modules including: Linear, Flatten, ReLU, Sequential, SoftmaxLoss, BatchNorm1d, LayerNorm1d, Dropout and Residual. Each module has init and forward functions. 
Notice: 
1. When the tensor is scalar, its dtype will change to float64 after compuation.
2. Numpy array should be converted to tensor first before computation with tensors.
3. Parameters should be initialized with Parameter() so that model.parameters() function can return all parameters of the model.

# Optimization
Notice:  
1. t should be increased outside for loop.
2. Weight decay should be done first and can't be used in final weight update.
3. Previous m and v aren't biased m and v.
4. It is better to detach grad and weight before computation.
5. The way to set updated parameter to model is to use p.data (which is used to change cached_data).

# Data
## Transforms
Current transforms can be applied to batch data.
## Dataset
Mnist dataset return 784 dim img. Do reshape before transforms.
## DataLoader
DataLoader should not only support to return (X, y). It should operate a tuple from dataset.

# MLP resnet
Notice:
1. Different places in the code to initialize the list of residual_blocks lead to different results.
2. Error rate should be calculated with all samples in the dataset and should not be calculated every batch first.

# Questions:
1. Why can't scalar tensor stay dtype?
2. Why does a tensor change type after computation with a numpy array?
3. Why do different places in the code to initialize the list of residual_blocks lead to different results?