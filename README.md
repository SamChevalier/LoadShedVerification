<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://i.imgur.com/bmX6RDK.png">
  <source media="(prefers-color-scheme: dark)" srcset="https://i.imgur.com/3KgHoMM.png">
  <img alt = "Grid graph for LoadShedVerification" src = "https://i.imgur.com/3KgHoMM.png">
</picture>

# LoadShedVerification 

Code associated with the paper "Maximal Load Shedding Verification for Neural Network Models of AC Line Switching". Code written by Duncan Starkenburg and Sam Chevalier. 


## Setup
To successfully pull this repo in its entirety to the local machine, you will need Git LFS (Large file storage).
Instructions on how to download Git LFS can be found [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).

On your local machine run the following code (NOTE: This repo is roughly 3GB; this may take a second):
```
$ git clone https://github.com/SamChevalier/LoadShedVerification
$ git lfs pull
```
Additionally, if for some reason the file pointers for the repo are glitched, (aka the last command did not pull the large files), you can manually pull:
```
$ git lfs fetch --all
$ git lfs checkout
```
Once you have this repository on your local machine, its time to prepare the dependencies.
Please make sure Julia 1.10.8 is installed, as this is the version of Julia in which this code was written.
```
julia> VERSION
v"1.10.8"
```
All julia library dependencies can be found [here](https://github.com/SamChevalier/LoadShedVerification/blob/7648608c2606c5754a6b2aaa46f4697e49521407/Project.toml).

A [Gurobi](https://www.gurobi.com/) license will be needed to create your own datasets; however, there are three already available in this repo.

Another small administrative task needed to run this pipeline from start to finish is Python3. We recommend creating a virtual environment dedicated to this repo. Instructions on creating a virtual environment can be found [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments).

Before running `model_gen.py` enter your created virtual environment. Make sure PyTorch is installed.
```
$ source /path/to/your_env_name/bin/activate
(your_env_name)$ pip install torch
```
later in the actual verificaiton process you will need to set [this global parameter](https://github.com/SamChevalier/LoadShedVerification/blob/7648608c2606c5754a6b2aaa46f4697e49521407/grid/run_verification_tests.jl#L17) to your created enviroment.

## Create a Datafile
Skip this and the next step (Solve Datafile with Gurobi) if you just want to use our datasets found here: [14bus](https://github.com/SamChevalier/LoadShedVerification/blob/7648608c2606c5754a6b2aaa46f4697e49521407/src/outputs/14_bus/data_file_14bus.h5), [24bus](https://github.com/SamChevalier/LoadShedVerification/blob/7648608c2606c5754a6b2aaa46f4697e49521407/src/outputs/24_bus/data_file_24bus.h5), [118bus](https://github.com/SamChevalier/LoadShedVerification/blob/7648608c2606c5754a6b2aaa46f4697e49521407/src/outputs/118_bus/data_file_118bus.h5).
