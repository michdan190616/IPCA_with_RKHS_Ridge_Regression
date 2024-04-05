# IPCA with RKHS Ridge Regression

### Machine Learning course (CS-433)

## General information 

This repository contains the report and code for the second project of Machine Learning course (CS-433).

## Team

Our team is composed by:
- Andrea Belvisi
- Michelangelo D'Andrea
- Matteo Ferrazzi

## Project

In this project we applied machine learning techniques to finance and especially to factor models using Instrumental Principal Component Analisys (IPCA) and Kernel regression. IPCA allows factor loadings to be a linear function of stocks' characteristics overcoming some limitations of standard factor model. Kernel regression further allows to have a non-linear relationship between factor loadings and stocks' characteristics. In the report, we present the final results of the analysis, as well as the methods we used to perform it.

## Dataset

The data can be downloaded from this link https://drive.switch.ch/index.php/s/MIN35MEq1fdz9kC, using the following password: MMAMLproject2!.
The data employed in this project comprises of a list of datasets indexed by date. Each date corresponds to a month from January 2000 to December 2020, and each dataset contains 94 different characteristics on n stocks as well as the returns. In our project we use just the top 100 stocks based on market cap and years from 2000 to 2020.

## Code structure 

- 'download_clean_data.py' contains the functions to download and clean the data.
- 'ipca.py' contains the funtions to run IPCA and IPCA regularized.
- 'kernel_regression.py' contains the functions that run kernel regressions and kernel regressions using low rank approximation.
- 'metrics.py' contains the functions to evaluate the performance of the models we implemented.
- 'validation.py' contains the functions to perform the validation of the models we implemented.
- 'validation_run.ipynb' contains the code to run the validation in order to obtain the best parameters for each model.
- 'run.ipynb' contains the code that produces the final results of our analysis.

## Other files

- 'report.pdf' contains our project report.
- 'dict_IPCA_reg.pickle' contains the result of the validation of IPCA regularized.
- 'dict_gaussian.pickle' contains the result of the validation of gaussian kernel regression.
- 'dict_rq.pickle' contains the result of the validation of rational quadratic kernel regression.

## Libraries 

- Numpy : used to manipulate arrays and matrices, and perform linear algebra operations.
- Pandas : used to manipulate datasets.
- Seaborn : used for visualization.
- Matplotlib : used for visualization.
- Importlib : used to import modules.
- Pickle : used to store the results of the validation in dictionaries.

## Usage

You need to clone this repository.

Then you need to put the content of the link from dataset section in a folder called Data to store in this repository. Consequently, you need to set the variable 'folder_path' in the code to the path of monthly_data (which is a folder contained in Data).

Finally you can run 'validation_run.ipynb' to get the optimal parameters, and 'run.ipynb' to get the results and plots.
It is also possible to run only 'run.ipynb', since we have already saved the dictionaries with optimal parameters.

'validation_run.ipynb' takes six hours to run while 'run.ipynb' takes an hour.



