import sys
import torch
import torch.nn as nn
import torch.optim as optim
from ExtinctionNetwork import ExtinctionNetwork
from FileHelper import FileHelper


class NetworkHelper:
    """
    A utility class for building and training neural networks for extinction and density estimation.

    # Methods:
        - `integral(tensor, network_model, min_distance=0., debug=0)`: Custom analytic integral of the network for
            MSE loss. (Static)
        - `init_weights(model)`: Initializes weights and biases using Xavier uniform initialization. (Static)
        - `create_net_integ(hidden_size)`: Creates a neural network and sets up the optimizer. (Static)
        
    # Example:
        The following example demonstrates how to use the `NetworkHelper` class to build and intit network weights.
        
        >>> network, optimizer = NetworkHelper.create_net_integ(hidden_size=128, device=device, learning_rate=0.001,
        >>>                                                    is_new_network=True, epoch_number=0,
        >>>                                                    config_file_path=config_file_path)
        >>> NetworkHelper.init_weights(network)
    """

    @staticmethod
    def integral(tensor, network_model, min_distance=0.):
        """
        Custom analytic integral of the network `ExtinctionNetwork` to be used in MSE loss.

        # Remarks:
            This function calculates a custom analytic integral of the network `ExtinctionNetwork`,
            as specified in the Equation 15a and Equation 15b of Lloyd et al. 2020,
            to be used in Mean Squared Error (MSE) loss during training.

        # Args:
            - `tensor (torch.Tensor)`: Input tensor of size (batch_size, 3).
            - `network_model (ExtinctionNetwork)`: The neural network model used for the integration.
            - `min_distance (float, optional)`: Minimum value for integration. Defaults to 0.

        # Returns:
            `torch.tensor`: Result of the custom analytic integral for each sample in the batch.
        """
        # TODO : Corriger doc
        # Equation 15b of Lloyd and al 2020 -> Phi_j for each neuron
        # Li_1(x) = -ln(1-x) for x \in C
        batch_size = tensor.size()[0]
        coord_num = tensor.size()[1]  # number of coordinates, the last one is the distance
        min_distance = tensor * 0. + min_distance

        a = -torch.log(1. + torch.exp(-1. * (network_model.linear1.bias.unsqueeze(1).expand(network_model.hidden_size,
                                                                                            batch_size) +
                                             torch.matmul(network_model.linear1.weight[:, 0:coord_num - 1],
                                                          torch.transpose(tensor[:, 0:coord_num - 1], 0, 1))
                                             )
                                      - torch.matmul(network_model.linear1.weight[:, coord_num - 1].unsqueeze(1),
                                                     torch.transpose(min_distance[:, coord_num - 1].unsqueeze(1), 0, 1)
                                                     )
                                      )
                       )
        b = torch.log(1. + torch.exp(-1. * (network_model.linear1.bias.unsqueeze(1).expand(network_model.hidden_size,
                                                                                           batch_size)
                                            + torch.matmul(network_model.linear1.weight[:, 0:coord_num - 1],
                                                           torch.transpose(tensor[:, 0:coord_num - 1], 0, 1)
                                                           )
                                            )
                                     - torch.matmul(network_model.linear1.weight[:, coord_num - 1].unsqueeze(1),
                                                    torch.transpose(tensor[:, coord_num - 1].unsqueeze(1), 0, 1)
                                                    )
                                     )
                      )

        phi_j = a + b

        # Equation 15a of Lloyd and al 2020: alpha_1=0, beta_1=x
        # Sum over all neurons of the hidden layer
        aa = network_model.linear2.bias * (tensor[:, coord_num - 1] - min_distance[:, coord_num - 1])

        bb = torch.matmul(network_model.linear2.weight[0, :],
                          (torch.transpose((tensor[:, coord_num - 1]
                                            - min_distance[:, coord_num - 1]).unsqueeze(1), 0, 1)
                           .expand(network_model.hidden_size, batch_size)
                           + torch.transpose(torch.div(torch.transpose(phi_j, 0, 1),
                                                       network_model.linear1.weight[:, coord_num - 1]),
                                             0, 1)
                           )
                          )

        result = aa + bb

        return result

    @staticmethod
    def init_weights(model):
        """
        Function to initialize weights and biases for the given PyTorch model.

        # Remarks:
            This function initializes the weights and biases of the linear layers in the model
            using Xavier (Glorot) uniform initialization for weights and sets bias values to 0.1.

        # Args:
            - `model (torch.nn.Module)`: The PyTorch model for which weights and biases need to be initialized.
        """
        if isinstance(model, nn.Linear):
            torch.nn.init.xavier_uniform_(model.weight)
            model.bias.data.fill_(0.1)

    @staticmethod
    def create_net_integ(hidden_size, device, learning_rate, is_new_network, epoch_number, config_file_path):
        """
        Function to create the neural network and set the optimizer.

        # Remarks:
            This function creates an instance of the `ExtinctionNetwork` neural network with the specified
            hidden size and initializes an Adam optimizer with the given learning rate.

        # Args:
            - `hidden_size (int)`: Size of the hidden layer in the neural network.
            - `device (torch.device)`: The device (CPU or GPU) on which the model is to be trained.
            - `learning_rate (float)`: The learning rate for the Adam optimizer.
            - `is_new_network (bool)`: Flag indicating whether to create a new network or load an existing one.
            - `epoch_number (int)`: The epoch number for which the network is being created.
            - `config_file_path (str)`: The path to the configuration file.

        # Returns:
            `tuple[ExtinctionNetwork, optim.Adam]`: A tuple containing the created neural network and
                the Adam optimizer.
        """
        if is_new_network:
            network = ExtinctionNetwork(hidden_size)
        else:
            network_file = FileHelper.give_config_value(config_file_path, "outfile") + f"_e{epoch_number}.pt"
            try:
                checkpoint = torch.load(network_file, map_location='cpu')
                network = ExtinctionNetwork(hidden_size)
                network.load_state_dict(checkpoint['integ_state_dict'])
                network.eval()
                network.to(device)
            except FileNotFoundError as e:
                print(e)
                print(f"The network does not exist. Be careful to the 'is_new_network' flag.")
                print("Maybe you have to set it to false without training the network before.\n Run the program again.")
                sys.exit()
            
        return network, optim.Adam(network.parameters(), lr=learning_rate)
