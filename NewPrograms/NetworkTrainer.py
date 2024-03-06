import torch.nn.functional as F
from NetworkHelper import NetworkHelper

class NetworkTrainer:
    """
    A class for training the extinction neural network.

    # Args:
        - `ext_loss_function (callable)`: Loss function for extinction.
        - `dens_loss_function (callable)`: Loss function for density.
        - `ext_reduction_method (str)`: Method for reducing the extinction loss.
        - `dens_reduction_method (str)`: Method for reducing the density loss.

    # Attributes:
        - `ext_loss_function (callable)`: Loss function for extinction.
        - `dens_loss_function (callable)`: Loss function for density.
        - `ext_reduction_method (str)`: Method for reducing the extinction loss.
        - `dens_reduction_method (str)`: Method for reducing the density loss.
        
    # Methods:
        - `take_step(in_batch, tar_batch, loss_ext_total, loss_dens_total, nu_ext,
            nu_dens)`: Performs one training step.
        - `validation(in_batch_validation_set, tar_batch_validation_set, nu_ext, nu_dens, val_ext_total,
            val_dens_total)`: Performs one validation step.
    """

    def __init__(self, network, device, opti, ext_loss_function, dens_loss_function, ext_reduction_method,
                 dens_reduction_method):
        self.network = network
        self.device = device
        self.opti = opti
        self.ext_loss_function = ext_loss_function
        self.dens_loss_function = dens_loss_function
        self.ext_reduction_method = ext_reduction_method
        self.dens_reduction_method = dens_reduction_method

    def take_step(self, in_batch, tar_batch, loss_ext_total, loss_dens_total, nu_ext, nu_dens):
        """
        Function to perform one training step.

        This function executes one training step for the neural network model.
        It updates the model's parameters based on the provided input (in_batch) and target (tar_batch) batches.
        The loss function is a combination of log likelihood loss for total extinction and
        mean squared error loss for density estimation.

        # Args:
            - `in_batch (torch.Tensor)`: Input batch for the neural network.
            - `tar_batch (torch.Tensor)`: Target batch for the neural network.
            - `loss_ext_total (float)`: Total loss for extinction.
            - `loss_dens_total (float)`: Total loss for density.
            - `nu_ext (float)`: Lagrange multiplier for extinction loss calculation.
            - `nu_dens (float)`: Lagrange multiplier for density loss calculation.
            
        # Raises:
            - `RuntimeError`: If the target batch has a different shape than expected.
            
        # Returns:
            `tuple[float, float]`: Total loss for extinction, Total loss for density.
        """

        tar_batch = tar_batch.float().detach()
        in_batch = in_batch.float().to(self.device)
        tar_batch = tar_batch.to(self.device)

        # copy in_batch to new tensor and sets distance to 0
        y0 = tar_batch.clone().detach()
        y0 = y0[:, 0].unsqueeze(1) * 0.

        # compute ANN prediction for integration
        # density estimation at each location
        dens = self.network.forward(in_batch)
        # total extinction : in_batch in [-1,1] after rescaling -> pass the lower
        # integration bound to the code as default is 0
        exthat = NetworkHelper.integral(in_batch, self.network, min_distance=-1.)

        # compute loss function for integration network 
        # total extinction must match observed value
        try:
            loss_extinction = nu_ext * self.ext_loss_function(exthat, tar_batch[:, 0],
                                                              reduction=self.ext_reduction_method
                                                              )
        except RuntimeError as e:
            if (f"The size of tensor a ({exthat.size()}) must match the size of tensor b ({tar_batch.size()}) "
                    "at non-singleton dimension 1") in str(e):
                print("This error is probably due to the fact that the target batch has a different shape "
                      "than expected. Please check the shape of the target batch and try again.")
                print("For more information please check the \"Important Note\" in the documentation of the"
                      " \"check_and_assign_loss_function\" method in the MainProgram")
                raise e
            else:
                print(e)
        # density at point in_batch must be positive
        loss_dens = nu_dens * self.dens_loss_function(F.relu(-1.*dens), y0, reduction=self.dens_reduction_method)

        # combine loss functions
        fullloss = loss_dens + loss_extinction

        # compute total loss of epoch (for monitoring)
        loss_ext_total += loss_extinction.item()
        loss_dens_total += loss_dens.item()

        # zero gradients before taking step (gradients are additive, if not set to zero then adds
        # to the previous gradients)
        self.opti.zero_grad()

        # compute gradients
        fullloss.backward()

        # do 1 optimisation step after minibatch
        self.opti.step()

        return loss_ext_total, loss_dens_total

    def validation(self, in_batch_validation_set, tar_batch_validation_set, nu_ext, nu_dens, val_ext_total,
                   val_dens_total):
        """
        Function to perform one validation step.

        This function executes one validation step for the neural network model.
        It evaluates the model's performance on the validation set based on the provided input (in_batch_validation_set)
        and target (tar_batch_validation_set) batches.
        The loss function is a combination of log likelihood loss for total extinction and
        mean squared error loss for density estimation.

        # Args:
            - `in_batch_validation_set (torch.Tensor)`: Input batch for the validation set.
            - `tar_batch_validation_set (torch.Tensor)`: Target batch for the validation set.
            - `nu_ext (float)`: Lagrange multiplier for extinction loss calculation.
            - `nu_dens (float)`: Lagrange multiplier for density loss calculation.
            - `val_ext_total (float)`: Total loss for extinction in the validation set.
            - `val_dens_total (float)`: Total loss for density in the validation set.
            
        # Raises:
            - `RuntimeError`: If the target batch has a different shape than expected.
            
        # Returns:
            `tuple[float, float]`: Total loss for extinction in the validation set, Total loss for density in the
                validation set.
        """
        tar_batch_validation_set = tar_batch_validation_set.float().detach()
        in_batch_validation_set = in_batch_validation_set.float().to(self.device)
        tar_batch_validation_set = tar_batch_validation_set.to(self.device)

        y0_val = tar_batch_validation_set.clone().detach()  # TODO delete ?
        y0_val = tar_batch_validation_set[:, 0].unsqueeze(1) * 0.

        # density estimation at each location
        dens = self.network.forward(in_batch_validation_set)

        # total extinction
        exthat = NetworkHelper.integral(in_batch_validation_set, self.network, min_distance=-1.)

        # compute loss function for  network : L2 norm
        # total extinction must match observed value
        try:
            loss_extinction = nu_ext * self.ext_loss_function(exthat, tar_batch_validation_set[:, 0],
                                                              reduction=self.ext_reduction_method
                                                              )
        except RuntimeError as e:
            if "The size of tensor a (500) must match the size of tensor b (2) at non-singleton dimension 1" in str(e):
                print("This error is probably due to the fact that the target batch has a different shape than"
                      " expected. Please check the shape of the target batch and try again.")
                print("For more information please check the \"Important Note\" in the documentation of the"
                      " \"check_and_assign_loss_function\" method in the MainProgram")
                raise e
            else:
                print(e)

        # density at point in_batch must be positive
        loss_density = nu_dens * self.dens_loss_function(F.relu(-1.*dens), y0_val, reduction=self.dens_reduction_method)

        val_dens_total += loss_density.item()
        val_ext_total += loss_extinction.item()

        return val_ext_total, val_dens_total
