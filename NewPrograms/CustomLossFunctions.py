class CustomLossFunctions:
    """
    A class containing custom loss functions for PyTorch models.

    # Methods:
        - loglike_loss(prediction, label, reduction_method): Computes the log likelihood loss. (Static)
    """
    @staticmethod
    def loglike_loss(prediction, label, reduction):
        """
        Function to compute the log likelihood loss.

        This function implements the log likelihood loss function. It assumes that 'label' is a pair (extinction, sigma)
        and returns <((x - label(E)) / label(sigma))**2>, where <.> is either the mean or the sum,
        depending on the 'reduction' parameter.

        # Args:
            - `prediction (torch.Tensor)`: Model predictions.
            - `label (torch.Tensor)`: Labels in the form (extinction, sigma).
            - `reduction (str)`: Method for reducing the loss.

        # Raises:
            - `Exception`: Raised if the reduction value is unknown. Should be 'sum' or 'mean'.

        # Returns:
            - `torch.Tensor`: Value of the log likelihood loss.
        """
        if reduction == 'sum':
            return (((prediction-label[:, 0])/label[:, 1])**2).sum()
        elif reduction == 'mean':
            return (((prediction-label[:, 0])/label[:, 1])**2).mean()
        else:
            raise Exception('reduction value unknown. Should be sum or mean')
