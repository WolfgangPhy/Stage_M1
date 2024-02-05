import torch.nn as nn

class ExtinctionNeuralNet(nn.Module):
    """
    Neural network model for extinction prediction.

    # Args:
        hidden_size (int): Number of hidden units in the neural network.

    # Attributes:
        hidden_size (int): Number of hidden units.
        linear1 (nn.Linear): First linear layer.
        linear2 (nn.Linear): Second linear layer.
        sigmoid (nn.Sigmoid): Sigmoid activation function.

    # Methods:
        - __init__(hidden_size): Initializes an instance of the ExtinctionNeuralNet class.
        - forward(tensor): Forward pass of the neural network.

    # Examples:
        >>> model = ExtinctionNeuralNet(hidden_size=128)
        >>> input_tensor = torch.tensor([1.0, 2.0, 3.0])
        >>> output = model(input_tensor)
        >>> print(output)
        tensor([0.1234], grad_fn=<AddmmBackward>)
    """
    def __init__(self, hidden_size):
        super(ExtinctionNeuralNet, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(3, self.hidden_size, bias=True)
        self.linear2 = nn.Linear(self.hidden_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, tensor):
        """
        Forward pass of the neural network.

        # Args:
            tensor (torch.Tensor): Input tensor of shape (batch_size, 3).

        # Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        out = self.linear1(tensor)
        out = self.sigmoid(out)
        out = self.linear2(out)
        return out