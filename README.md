# Instructions for how to navigate this repository

This project is the exam assignment for the ACIT4100 class at OsloMet. 
The repository contains the code of a TSK estimation implementation. The following files are important:

- The file used to produce the training and test is the preliminary_analysis.ipynb. It is important that this notebook is executed whenever the other parts of the project are tested.
- The TSK fuzzy controller is defined in the file: tsk_model.py. If the file is executed, a simple TSK model is defined. This part was used for testing under the development of the TSK code. A 3D-plot describing the model is showed.
- The methods for deriving the TSK from a dataset is found in the TSKModel class in tsk_model_py.
- The Bees Algorithm implemented for fine tuning the TSK model can be found in the file ba_optimizatino.py. This file can be executed to test the optimization. The output will be the evolving rmse values.
- The evaluation of the tsk models were done in the file evaluation.ipynb. By running this notebook the table and plot of the evaluation can be produced.
- Testing of the Bees algorithm can be found in test_bees_optimization.ipynb.
- The Matlab ANFIS implementation can be found in Anfis_matlab_1.mlx

The dataset used in this project is available here:
- https://doi.org/10.6084/m9.figshare.12155553

Please make sure the csv containing the dataset is located in the dataset folder.

### Create the environment
To keep track of dependencies an environment management system was used: venv. Please make sure to create a venv environment and install the requirements by following these steps:

- Open the terminal from the root folder of the github repository (assuming the repository is cloned to your local computer).

- Run the command: python -m venv test_env to create the virtual environment.

- Activate the environment:
    - Mac: source test_env/bin/activate 

    - Windows: ./test_env/Script/activate

- Install libraries: pip install -r requirements.txt

To make the environment available in jupyter, a ipykernel needs to be installed: python -m ipykernel install --name=test_env

### Running CI_RF_mm in jupyter
The code can be run as is simply hitting shift+enter will run the selected cell and select the next, repeat until there are no more code segments. 
any cells with only df or df.info() can be rerun without causing any error since it is only used to display current df.
