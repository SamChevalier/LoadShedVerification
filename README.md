<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://i.imgur.com/bmX6RDK.png">
  <source media="(prefers-color-scheme: dark)" srcset="https://i.imgur.com/3KgHoMM.png">
  <img alt = "Grid graph for LoadShedVerification" src = "https://i.imgur.com/3KgHoMM.png">
</picture>

# LoadShedVerification 

Code associated with the paper "Maximal Load Shedding Verification for Neural Network Models of AC Line Switching". Code written by Duncan Starkenburg and Sam Chevalier. 


## Setup
To successfully pull this repository in its entirety to the local machine, you will need Git LFS (Large File Storage).
Instructions on how to download Git LFS can be found [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).

On your local machine, run the following code (NOTE: This repo is roughly 3GB; this may take a second):
```
$ git clone https://github.com/SamChevalier/LoadShedVerification
$ git lfs pull
```
Additionally, if for some reason the file pointers for the repo are glitched (aka the last command did not pull the large files), you can manually pull:
```
$ git lfs fetch --all
$ git lfs checkout
```
Once you have this repository on your local machine, it's time to prepare the dependencies.
Please make sure Julia 1.10.8 is installed, as this is the version of Julia in which this code was written.
```
julia> VERSION
v"1.10.8"
```
All Julia library dependencies can be found [here](https://github.com/SamChevalier/LoadShedVerification/blob/7648608c2606c5754a6b2aaa46f4697e49521407/Project.toml).

A [Gurobi](https://www.gurobi.com/) license will be needed to create your own datasets; however, there are three already available in this repo.

Another small administrative task needed to run this pipeline from start to finish is Python3. We recommend creating a virtual environment dedicated to this repo. Instructions on creating a virtual environment can be found [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments).

Before running [`model_gen.py`](https://github.com/SamChevalier/LoadShedVerification/blob/585bc1fa21cbcddc98cc67ce0f4c3fa2c0db33b5/src/model_gen.py) enter into your previously created virtual environment and confirm that PyTorch is installed:
```
$ source /path/to/your_env_name/bin/activate
(your_env_name)$ pip install torch
```
Later in the actual verification process, you will need to set [this global parameter](https://github.com/SamChevalier/LoadShedVerification/blob/7648608c2606c5754a6b2aaa46f4697e49521407/grid/run_verification_tests.jl#L17) to point to your virtual environment.

## Create a Datafile
*Skip this and the next step (Solve Perturbed Cases with Gurobi) if you just want to use our datasets found here: [14bus](https://github.com/SamChevalier/LoadShedVerification/blob/7648608c2606c5754a6b2aaa46f4697e49521407/src/outputs/14_bus/data_file_14bus.h5), [24bus](https://github.com/SamChevalier/LoadShedVerification/blob/7648608c2606c5754a6b2aaa46f4697e49521407/src/outputs/24_bus/data_file_24bus.h5), [118bus](https://github.com/SamChevalier/LoadShedVerification/blob/7648608c2606c5754a6b2aaa46f4697e49521407/src/outputs/118_bus/data_file_118bus.h5).*

The following file is used to create an HDF5 file of perturbed case data from the Optimal Power Flow benchmark library provided by the IEEE. In the context of this code, alpha is a scalar value that represents 'overall risk weight'. Essentially, 0 means try and keep all the lines on, 1 means try and turn off all the lines.

In the [`create_datafile.jl`](https://github.com/SamChevalier/LoadShedVerification/blob/585bc1fa21cbcddc98cc67ce0f4c3fa2c0db33b5/src/create_datafile.jl) file, there are several parameters, all at the top of the file.

[`model_name`](https://github.com/SamChevalier/LoadShedVerification/blob/d2ef45b7edf8eedc7f968d5d2b39e1629d4d5dd3/src/create_datafile.jl#L8): Should be set to the PGLib model you want to verify over. More info about PGLib.jl can be found [here](https://github.com/noahrhodes/PGLib.jl).

[`h5write_filename`](https://github.com/SamChevalier/LoadShedVerification/blob/d2ef45b7edf8eedc7f968d5d2b39e1629d4d5dd3/src/create_datafile.jl#L11): The name/path declaration for the file output of this code.

[`n_data`](https://github.com/SamChevalier/LoadShedVerification/blob/d2ef45b7edf8eedc7f968d5d2b39e1629d4d5dd3/src/create_datafile.jl#L14): How many perturbed datasets you wish to create

[`alpha_min`](https://github.com/SamChevalier/LoadShedVerification/blob/8ec39387af24e50817c32abb8bf1983eb4abb80f/src/create_datafile.jl#L17): The minimum value for a randomly generated 'alpha' during perturbation. 

[`alpha_max`](https://github.com/SamChevalier/LoadShedVerification/blob/8ec39387af24e50817c32abb8bf1983eb4abb80f/src/create_datafile.jl#L18): The maximum value for a randomly generated 'alpha' during perturbation.

[`perturb_percent`](https://github.com/SamChevalier/LoadShedVerification/blob/8ec39387af24e50817c32abb8bf1983eb4abb80f/src/create_datafile.jl#L19): The percent perturbation desired of the base case. (1 +- perturb_percent, i.e. perturb_percent = 0.75 then 0.25 - 1.75 * each aspect of the network)

After running this file, you will have an HDF5 file with `n_data` sample cases with each case having a random alpha value between `alpha_min` and `alpha_max`, and every qd and pd value will be perturbed by 1 +- `perturb_percent`. At the moment, per-line power risk is randomly assigned a value between 0 and 1, and scaled by alpha.

## Solve Perturbed Cases with Gurobi
*To reitterate: A [Gurobi](https://www.gurobi.com/) license will be needed to create your own datasets; however, there are three already available in this repo.*

The following file is used to find the optimal power-flow shutoff of each case using Gurobi. It uses the file created in the previous step and adds the solution dataset (branch decisions) alongside the sample in the HDF5 file.

The top of the [`gurobi_solve_datafile.jl`](https://github.com/SamChevalier/LoadShedVerification/blob/8ec39387af24e50817c32abb8bf1983eb4abb80f/src/gurobi_solve_datafile.jl) file, like the previous file, has several parameters.

NOTE: Although rare, if the program were to quit while Julia still has the HDF5 open, the file will be corrupted and become inaccessible. Please backup your file before running this program!

[`model_name`](https://github.com/SamChevalier/LoadShedVerification/blob/8ec39387af24e50817c32abb8bf1983eb4abb80f/src/gurobi_solve_datafile.jl#L14): Name of the PGlib case you used in the last step (should match).

[`output_file`](https://github.com/SamChevalier/LoadShedVerification/blob/8ec39387af24e50817c32abb8bf1983eb4abb80f/src/gurobi_solve_datafile.jl#L16): Path to the file created in the previous step (should match h5write_filename).

[`mip_gap`](https://github.com/SamChevalier/LoadShedVerification/blob/8ec39387af24e50817c32abb8bf1983eb4abb80f/src/gurobi_solve_datafile.jl#L18): A proportion (representing a percentage) that Gurobi will use as the optimal cutoff MIPGap when solving.

[`number_to_solve`](https://github.com/SamChevalier/LoadShedVerification/blob/8ec39387af24e50817c32abb8bf1983eb4abb80f/src/gurobi_solve_datafile.jl#L27): How many cases in your file you want to solve during this execution of the program, -1 means keep solving until finished. Useful when wanting to solve all the data over the course of different days, etc. The file will always pick up where you left off and continue solving.

Additionally, it should be noted that a less accurate but faster solving function can be implemented [on line 51](https://github.com/SamChevalier/LoadShedVerification/blob/9946ba2bb73e656d1f520fbaee09b796ad6e73a9/src/gurobi_solve_datafile.jl#L51). You can replace it with any of the `solve_` functions in [THIS](https://github.com/noahrhodes/LinearSOC/blob/main/src/prob.jl) repo.

After running this file, you will have a completed dataset stored in an HDF5 file. The structure of the file will be:

data_file_XXbus.h5:<br>
&nbsp;&nbsp;&nbsp;&nbsp;- alpha_min <sub><sup>*float*</sub></sup><br>
&nbsp;&nbsp;&nbsp;&nbsp;- alpha_max <sub><sup>*float*</sub></sup><br>
&nbsp;&nbsp;&nbsp;&nbsp;- perturb_percent <sub><sup>*float*</sub></sup><br>
&nbsp;&nbsp;&nbsp;&nbsp;- sample_data: <sub><sup>*group*</sub></sup><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- "index": <sub><sup>*int*</sub></sup><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- "num_samples": <sub><sup>*int*</sub></sup><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- "1": <sub><sup>*group*</sub></sup><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- "2": <sub><sup>*group*</sub></sup><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-  ...<br>
&nbsp;&nbsp;&nbsp;&nbsp;- total_samples <sub><sup>*int*</sub></sup><br>

Where each sample is stored as follows, where **status** is the solution array of 0s or 1s; 1 being a line kept on:

"1":<br>
&nbsp;&nbsp;&nbsp;&nbsp;- load <sub><sup>*group*</sub></sup><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- qd <sub><sup>*float array*</sub></sup><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- pd <sub><sup>*float array*</sub></sup><br>
&nbsp;&nbsp;&nbsp;&nbsp;- branch <sub><sup>*group*</sub></sup><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- power_risk <sub><sup>*float array*</sub></sup><br>
**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- status** <sub><sup>*float array*</sub></sup><br>
&nbsp;&nbsp;&nbsp;&nbsp;- alpha <sub><sup>*float*</sub></sup><br>

## Creating a PyTorch Model
*Our pretrained PyTorch models that were used in the paper can be found here: [14bus](https://github.com/SamChevalier/LoadShedVerification/tree/8ec39387af24e50817c32abb8bf1983eb4abb80f/src/outputs/14_bus), [24bus](https://github.com/SamChevalier/LoadShedVerification/tree/8ec39387af24e50817c32abb8bf1983eb4abb80f/src/outputs/24_bus), [118bus](https://github.com/SamChevalier/LoadShedVerification/tree/8ec39387af24e50817c32abb8bf1983eb4abb80f/src/outputs/118_bus). For any other case, or to try it yourself, keep reading :)*

The following file takes your optimally solved PGLib cases stored in the HDF5 file created above and outputs a trained, fully connected neural network in PyTorch. It will also output an HDF5 file with the calculated normalization values of the dataset, which you will use for pre-processing during verification.

The top of the [`model_gen.py`](https://github.com/SamChevalier/LoadShedVerification/blob/8ec39387af24e50817c32abb8bf1983eb4abb80f/src/model_gen.py) file, like the two previous files, has several parameters.

[`epochs`](https://github.com/SamChevalier/LoadShedVerification/blob/b4b51ca2966210c4803cfa6f444ec0b96d3ff37e/src/model_gen.py#L13): The number of Epochs you wish to train the neural network over.

[`hidden_dim_depth`](https://github.com/SamChevalier/LoadShedVerification/blob/b4b51ca2966210c4803cfa6f444ec0b96d3ff37e/src/model_gen.py#L15): The number of nodes (neurons) in the two hidden, connected layers. *We trained at 32, 128, 512, and 2048 for each case*

[`batch_sizes`](https://github.com/SamChevalier/LoadShedVerification/blob/b4b51ca2966210c4803cfa6f444ec0b96d3ff37e/src/model_gen.py#L17): How many samples per training batch. *We used 5 for our 14 and 24 cases, and 1 for our 118 case*

[`dropout_percent`](https://github.com/SamChevalier/LoadShedVerification/blob/b4b51ca2966210c4803cfa6f444ec0b96d3ff37e/src/model_gen.py#L19): Proportion (representing a percentage) of data to drop during training to prevent overfitting. *We trained with 0.2 (20%).*

[`learn_rate`](https://github.com/SamChevalier/LoadShedVerification/blob/b4b51ca2966210c4803cfa6f444ec0b96d3ff37e/src/model_gen.py#L21): Step size for learning during training, passed to `torch.optim.Adam` optimizer. *We trained with 1e-4*

[`output_filename`](https://github.com/SamChevalier/LoadShedVerification/blob/b4b51ca2966210c4803cfa6f444ec0b96d3ff37e/src/model_gen.py#L23): In hindsight, this should probably be named input_filename for this program; However, it should MATCH the `output_file` from the previous step. (HDF5 with samples)

[`normalization_filename`](https://github.com/SamChevalier/LoadShedVerification/blob/b4b51ca2966210c4803cfa6f444ec0b96d3ff37e/src/model_gen.py#L24): Path/name for HDF5 file that will contain the standard deviations and means across the dataset for every pd and qd value. **These values are already corrected for divide-by-zero during normalization by adding 1e-6 to each standard deviation.**

[`percent_train`](https://github.com/SamChevalier/LoadShedVerification/blob/b4b51ca2966210c4803cfa6f444ec0b96d3ff37e/src/model_gen.py#L26): Proportion (representing a percentage) of samples to be used in training. *We trained with 0.8*

[`percent_val`](https://github.com/SamChevalier/LoadShedVerification/blob/b4b51ca2966210c4803cfa6f444ec0b96d3ff37e/src/model_gen.py#L27): Proportion (representing a percentage) of samples to be used in validation. *We trained with 0.1*

[`percent_test`](https://github.com/SamChevalier/LoadShedVerification/blob/b4b51ca2966210c4803cfa6f444ec0b96d3ff37e/src/model_gen.py#L28): Proportion (representing a percentage) of samples to be used in testing. *We trained with 0.1*

NOTE: `percent_train`, `percent_val`, and `percent_test` should sum to 1 or an error will be thrown during execution!

After this file runs, you will be given a .png showing basic training data and .pt file of your model with the following structure:

```
model = Sequential(
            Linear(<input_size> => <hidden_dim_depth>)
            ReLU
            Dropout(<dropout_percentage>)
            Linear(<hidden_dim_depth> => <hidden_dim_depth>)
            ReLU
            Dropout(<dropout_percentage>)
            Linear(<hidden_dim_depth> => <hidden_dim_depth>)
            ReLU
            Dropout(<dropout_percentage>)
            Linear(<hidden_dim_depth> => <number_of_branch_decisions>)
```
NOTE: The Dropout layers are removed before saving the model!
