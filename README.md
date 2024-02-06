# MolBindDif

## Installation
We recommend miniconda (or anaconda). Run the following to install a conda environment with the necessary dependencies.
```
conda env create -f mbd.yml
```
Next, we recommend installing our code as a package. To do this, run the following.
```
pip install -e .
```
One will also need to install Voronota package and add it to the ```PATH```.
## Inference
For now, the inference can only be performed on complexes from [PPI3D](http://bioinformatics.ibt.lt/ppi3d/) and with MolBindDif-ba model.
The command to run the inference is
```
python run_inference.py
```
By default, the inference will run on the *protein_nucleic-6hcf-1-6hcf_J3-1-6hcf_72-1* sample. If you want to run MolBindDif on some complexes of your choice, you should create a .csv file with their names in column ```'ppi3d_name'``` and insert its path in ```./config/inference.yaml``` file in ```input_file``` field.
