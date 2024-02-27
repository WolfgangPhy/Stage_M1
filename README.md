# Galaxy mapping using perceptron neural network

# Description

The aim of this project is to test the ability of a perceptron or multi-layer perceptron neural network to invert a linear Volterra equation of the first kind which is the equation that relates the absorption density  to the total density along lines of sight to astronomical objects. This inversion process is well known to be ill-conditioned as measurement errors can lead to unphysical results such as negative densities.

This code is build to allow the testing of some loss functions and parameters (by using the 'Parameters.json' file) in order to find the good combination of parameters and loss function to obtain the best convergence results and possibly apply it to real data as the Gaia catalog.

# Installation

To install this code you can clone the following repository : https://github.com/WolfgangPhy/

```bash
git clone https://github.com/WolfgangPhy/
```
# Documentation

The documentation of the code is available in the `Documentation` directory. You can find it in two fomat, in pdf and html. The html format is more user friendly and you can navigate through the differents classes and methods easily. To use it open the `index.html` file in your web browser.

# How it works ?

The code is build in a way that almost all the parameters are tunable in the 'Parameters.json" file, the only thing that need your attention is when you want to change the loss function, some of it takes differents parameters than the others (more details in the 'Important Note' section of the `check_and_assign_loss_function` method documenation).
When you run the code (see *How to run the code ?* part), the code automatically generate a directory named with the parameters used in the 'Parameters.json' file. In this directory, you will see the following files and directories:
- The 'Parameters.json' file that contains the parameters used for the run (a copy of the original file that is in the `Source` directory).
- The 'Config.json' file that contains the the path for differents files (for the current test) used in the code (if you want to see how it works, you can check the `FileHelper` class documentation).
- The 'PyTorchFiles' directory that contains the dataset 'fiducial_model.pt' and the trained models at different epochs 'ExctinctionModelMean_e{epoch_number}.pt'
- The 'OutputFiles' directory that contains the loss and the validation datas in .csv format and a log file that gives some additional informations about the current test.
- The 'NpzFiles' directory that contains datas about the extinction and density in .npz format, there is two files for both extinction and density, one for the data on a 2D grid and one along some lines of sight.
- The 'Plots' directory that contains the plots genereated by the `Visualizer` class.

# Structure of the code

This code was build using the Single responsibility principle, the code is divided in differents classes (one class by file), each of it has a specific role (more infos in the documentation of each class)

# How to run the code ?

To run the code, you need to have the following packages installed:
- numpy
- torch
- matplotlib
- pandas
- multiprocessing
- json
- os
- seaborn
- math
- scipy
- pickle
- shutil
- sys
- csv
- tqdm

Then you need to run the `execute()` method of the `MainProgram` class. Given that this lethod is not static, you need to create an instance of the `MainProgram` class and then call the `execute()` method. For exemple you can do it like this :
    
```Python
    mainprogram = MainProgram()
    mainprogram.execute()
```

Ensure that the 'Parameters.json'  and the "Config.json" files are in the `Source` directory.

# Limitations

This code is a solid base but has some limitations, here are some of them:
- The code can have some speed optimisation, especially the in the `ExtinctionModelHelper` and the `ModelCalculator`classes.
- The code is not able to compute on 3 dimensions, it can be a good improvement to add this feature.
- Some of the parameters (not the important ones but for exemple the boundaries of the map) are hard coded in the code, it can be a good improvement to add them in the 'Parameters.json' file.

# Parameters 

The 'Parameters.json' file looks like this :

```json

{
    "nu_ext": 1.0,
    "nu_dens": 1.0,
    "ext_loss_function": "mse_loss",
    "ext_loss_function_custom": false,
    "dens_loss_function": "mse_loss",
    "dens_loss_function_custom": false,
    "star_number": 10000,
    "ext_reduction_method": "mean",
    "dens_reduction_method": "sum",
    "epoch_number": 8000,
    "learning_rate": 0.001,
    "batch_size": 500,
    "compute_density" : false
}
```

The parameters are the following:
- nu_ext: Lagrange multiplier for extinction loss calculation.
- nu_dens: Lagrange multiplier for density loss calculation.
- ext_loss_function: The loss function used for the extinction
- ext_loss_function_custom: A boolean that indicates if the loss function is custom or not (if it is custom you need to take a function in the`CustomLossFunction` class, be careful, some of the loss function takes differents parameters than the others, more details in the `check_and_assign_loss_function` method documenation)
- dens_loss_function: The loss function used for the density
- dens_loss_function_custom: A boolean that indicates if the loss function is custom or not
- star_number: The number of stars used to generate the extinction and density
- ext_reduction_method: The method used to reduce the extinction
- dens_reduction_method: The method used to reduce the density
- epoch_number: The number of epochs used for the training
- learning_rate: The learning rate (this is a base value, the optimizer can change it during the training)
- batch_size: The size of sub-batches used for the training
- compute_density: A boolean that indicates if the density is computed or not (this parameter is used to do calculation on the extinction only and so reduce the running time)
