import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ExtinctionNeuralNet as NeuralNet

class ExtinctionNeuralNetBuilder:
    """
    A utility class for building and training neural networks for extinction and density estimation.

    # Args:
        `device (torch.device)`: Device on which the neural network is running.
        `hidden_size (int)`: Size of the hidden layer in the neural network.
        `learning_rate (float)`: Learning rate for the Adam optimizer.

    # Attributes:
        `network (ExtinctionNeuralNet)`: The neural network model for extinction and density estimation.
        `opti (optim.Adam)`: The Adam optimizer used for training.
        `device (torch.device)`: Device on which the neural network is running.
        `learning_rate (float)`: Learning rate for the Adam optimizer.

    # Methods:
        - `integral(tensor, network_model, xmin=0., debug=0)`: Custom analytic integral of the network for MSE loss.
        - `init_weights(model)`: Initializes weights and biases using Xavier uniform initialization.
        - `create_net_integ(hidden_size)`: Creates a neural network and sets up the optimizer.
        
    # Example:
        >>> # Example usage of ExtinctionNeuralNetBuilder
        >>> hidden_size = 64
        >>> learning_rate = 0.001

        >>> # Create an instance of ExtinctionNeuralNetBuilder
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> builder = ExtinctionNeuralNetBuilder(device, hidden_size, learning_rate)
    """
    
    def __init__(self, device, hidden_size, learning_rate):
        self.device = device
        self.learning_rate = learning_rate
        self.network, self.opti = self.create_net_integ(hidden_size)
              
    def integral(self, tensor, network_model, distance_min=0., debug=0):
        """
        Custom analytic integral of the network ExtinctionNeuralNet to be used in MSE loss.

        This function calculates a custom analytic integral of the network ExtinctionNeuralNet,
        as specified in the Equation 15a and Equation 15b of Lloyd et al. 2020,
        to be used in Mean Squared Error (MSE) loss during training.

        # Args:
            - `tensor (torch.Tensor)`: Input tensor of size (batch_size, 3).
            - `network_model (ExtinctionNeuralNet)`: The neural network model used for the integration.
            - `xmin (float, optional)`: Minimum value for integration. Defaults to 0.
            - `debug (int, optional)`: Debugging flag. Defaults to 0.

        # Returns:
            - `torch.tensor`: Result of the custom analytic integral for each sample in the batch.
        """
        # Equation 15b of Lloyd et al 2020 -> Phi_j for each neuron
        # Li_1(x) = -ln(1-x) for x \in C
        batch_size = tensor.size()[0]
        coord_num = tensor.size()[1]  # number of coordinates, the last one is the distance
        distance_min = tensor * 0. + distance_min

        a = -torch.log(1. + torch.exp(-1. * (network_model.linear1.bias.unsqueeze(1).expand(network_model.hidden_size, batch_size)
                                                + torch.matmul(network_model.linear1.weight[:, 0:coord_num - 1], 
                                                torch.transpose(tensor[:, 0:coord_num - 1], 0, 1))\
                                            )
                                        - torch.matmul(network_model.linear1.weight[:, coord_num - 1].unsqueeze(1), 
                                                            torch.transpose(distance_min[:, coord_num - 1].unsqueeze(1), 0, 1)
                                                        )
                                    )
                    )
        b = torch.log(1. + torch.exp(-1. * (network_model.linear1.bias.unsqueeze(1).expand(network_model.hidden_size, batch_size)
                                                + torch.matmul(network_model.linear1.weight[:, 0:coord_num - 1], 
                                                torch.transpose(tensor[:, 0:coord_num - 1], 0, 1))
                                            )
                                        - torch.matmul(network_model.linear1.weight[:, coord_num - 1].unsqueeze(1),
                                                            torch.transpose(tensor[:, coord_num - 1].unsqueeze(1), 0, 1)
                                                            )
                                        )
                        )

        phi_j = a + b

        # Equation 15a of Lloyd et al 2020: alpha_1=0, beta_1=x
        # Sum over all neurons of the hidden layer
        aa = network_model.linear2.bias * (tensor[:, coord_num - 1] - distance_min[:, coord_num - 1])

        bb = torch.matmul(network_model.linear2.weight[0, :],
                            (torch.transpose( (tensor[:, coord_num - 1] - distance_min[:, coord_num - 1])
                                            .unsqueeze(1), 0, 1)
                                .expand(network_model.hidden_size, batch_size)
                                + torch.transpose(torch.div(torch.transpose(phi_j, 0, 1), 
                                                            network_model.linear1.weight[:, coord_num - 1]),
                                                0, 1)
                                )
                            )

        result = aa + bb

        return result

    def init_weights(self, model):
        """
        Function to initialize weights and biases for the given PyTorch model.

        This function initializes the weights and biases of the linear layers in the model
        using Xavier (Glorot) uniform initialization for weights and sets bias values to 0.1.

        # Args:
            - `model (torch.nn.Module)`: The PyTorch model for which weights and biases need to be initialized.
        """
        if type(model) == nn.Linear:
            torch.nn.init.xavier_uniform_(model.weight)
            model.bias.data.fill_(0.1)
            
    def create_net_integ(self, hidden_size):
        """
        Function to create the neural network and set the optimizer.

        This function creates an instance of the ExtinctionNeuralNet neural network with the specified
        hidden size and initializes an Adam optimizer with the given learning rate.

        # Args:
            - `hidden_size (int)`: Size of the hidden layer in the neural network.

        # Returns:
            - `tuple[ExtinctionNeuralNet, optim.Adam]`: A tuple containing the created neural network and the Adam optimizer.
        """
        network = NeuralNet.ExtinctionNeuralNet(hidden_size, self.device)
        return network, optim.Adam(network.parameters(), lr=self.learning_rate)

    
