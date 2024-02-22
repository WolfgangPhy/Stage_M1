

class CustomLossFunctions():
    """
    A class containing custom loss functions for PyTorch models.

    Methods:
    - loglike_loss(prediction, label, reduction_method): Computes the log likelihood loss. (Static)

    Usage Example:
    >>> prediction = model(input_data)
    >>> loss = CustomLossFunctions.loglike_loss(prediction, labels, reduction_method='mean')
    >>> loss.backward()

    """
    @staticmethod
    def loglike_loss(prediction, label, reduction_method):
        """
        Function to compute the log likelihood loss.

        This function implements the log likelihood loss function. It assumes that 'label' is a pair (E, sigma)
        and returns <((x - label(E)) / label(sigma))**2>, where <.> is either the mean or the sum, depending on the 'reduction' parameter.

        # Args:
            - `prediction (torch.Tensor)`: Model predictions.
            - `label (torch.Tensor)`: Labels in the form (E, sigma).
            - `reduction_method (str)`: Method for reducing the loss.

        # Raises:
            - `Exception`: Raised if the reduction value is unknown. Should be 'sum' or 'mean'.

        # Returns:
            - `torch.Tensor`: Value of the log likelihood loss.
        """
        if reduction_method=='sum':
            return ( ( ( prediction-label[:,0] )/label[:,1] )**2 ).sum()
        elif reduction_method=='mean':
            return ( ( ( prediction-label[:,0] )/label[:,1] )**2 ).mean()
        else :
            raise Exception('reduction value unkown. Should be sum or mean')