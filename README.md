# Highway-Network-PyTorch
PyTorch Implementation of [Training Very Deep Networks](https://arxiv.org/abs/1507.06228) 
by Rupesh K Srivastava, Klaus Greff and  Jürgen Schmidhuber.

To initialize the network, you need to specify it's size, depth and non-linear activation function. 
With something like:

```python
Highway(
    size=512, 
    num_layers=20, 
    activation_fctn=torch.nn.ReLU, 
    drop_out=0.2
)
``` 
