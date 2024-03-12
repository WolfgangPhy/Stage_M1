import torch.nn as nn


class ExtinctionNetwork(nn.Module):
    """
    Neural network model for extinction and density predictions.

    # Args:
        - `hidden_size (int)`: Number of hidden units in the neural network.
        - `device (torch.device)`: Device to run the neural network on.

    # Attributes:
        - `hidden_size (int)`: Number of hidden units.
        - `linear1 (nn.Linear)`: First linear layer.
        - `linear2 (nn.Linear)`: Second linear layer.
        - `sigmoid (nn.Sigmoid)`: Sigmoid activation function.

    # Methods:
        - `forward(tensor)`: Forward pass of the neural network.

    # Examples:
        The following example shows how to do a forward pass using the ExtinctionNetwork class.
        
        >>> network = ExtinctionNetwork(hidden_size=128)
        >>> network.to(device)
        >>> input_tensor = torch.randn(1, 3, device=device)
        >>> output = network(input_tensor)
        
        Or using forward explicitly:
        
        >>> output = network.forward(input_tensor)
    """
    def __init__(self, hidden_size):
        super(ExtinctionNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(3, self.hidden_size, bias=True)
        self.linear2 = nn.Linear(self.hidden_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, tensor):
        """
        Forward pass of the neural network.
        
        # Remarks:
            This method can be called implicitly by passing the input tensor to the network.

        # Args:
            - `tensor (torch.Tensor)`: Input tensor of shape (batch_size, 3).

        # Returns:
            `torch.Tensor`: Output tensor of shape (batch_size, 1).
            
        """
        out = self.linear1(tensor)
        out = self.sigmoid(out)
        out = self.linear2(out)
        return out
