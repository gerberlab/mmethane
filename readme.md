
## Installation
It is recommended to install MMETHANE and all requirements in a virtual environment. To create
and activate a virtual environment with Python 3.11 (recommended), run 
> conda create -n <env-name> python=3.11 \
> conda activate <env-name>

You can now install MMETHANE by either using pip or cloning this repository. To install with pip, run:
>pip install mmethane==0.1

If you prefer to clone this repository, run 
> git clone https://github.com/gerberlab/mmethane.git \
> cd mmethane

Next, install all requirements needed to run MMETHANE. The easiest way to do this is to download the "requirements.txt" 
file in this repository and run:
> pip install -r requirements.txt \
> conda install -c etetoolkit ete3

## Tutorial
To try out MMETHANE, you can use the sample config file provided in **config_files/sample.cfg**

If you have installed MMETHANE with pip, run:

>mmethane -c config_files/sample.cfg -o <absolute/path/to/output/folder/>

(replacing <absolute/path/to/output/folder/> with the **absolute** path to your output folder)

If you have cloned the github repository, run
> cd mmethane \
> python3 ./mmethane/run.py -c config_files/sample.cfg -o logs/

Running this command will (1) process an example dataset, (2) run MMETHANE, and (3) output an html file with visualizations

The HTML visualization will be located at "<output/folder>/mmethane_franzosa/seed_0/visualization.html"

To run lasso logistic regression with the dataset previously processed, run the same commands as above but replace "sample.cfg" with "sample_LR.cfg"
For instance:
> python3 ./mmethane/run.py -c config_files/sample.cfg -o logs/

## Creating a config file for your dataset. 
To use MMETHANE with your own data, simply create a .config file (or modify "sample.cfg") as outlined below. Required input arguments are highlighted.
> <mark>**[description]**<mark/> \
> <mark>**tag**:*name to be used for folder containing processed data and run results (if "run_name" not specified in [run])*</mark>
>
> **in_path**:*(optional) path to folder containing all input data. If used, you can use ${description:in_path} below when 
> input data files within this path*
>
> <mark>**[data]**</mark>\
> <mark>**subject_data**:*Full path to subject meta-data file, as .csv or .tsv. 
> Data should have samples as the rows and the outcome as a column. 
> If data is to be split on a coviariate, the covariate should also be included as a column.*</mark>
>
> <mark>**outcome_variable**:*Column name in subject_data that has subject outcomes*</mark>
>
> **outcome_positive_value**:*(optional) If outcome values are not ints (0s and 1s), specify the outcome string that 
> signifies the outcome has occured*
> 
> **outcome_negative_value**:*(optional) Can specify instead of outcome_positive_value or can specify both. 
> Specifying both is useful if data has some samples with a third outcome; these samples won't be included in processed data*
> 
> **sample_id_column**:*(optional) Use to signify the sample IDs column in subject_data if the sample IDs are not the 
> index of subject_data*
> 
> **covariate_variable**:*(optional) If needed, specify covariate variable that data should be split by*
> 
> **covariate_positive_value**:*(optional) If specifying covariate_variable and covariate values are not ints (0s,1s), 
> specify covariate positive value*
> 
> **covariate_negative_value**:*(optional) Can specify instead of outcome_positive_value or can specify both (i.e. if 
> some samples have third covariate that shouldn't be in processed data)*
>
> <mark>**[run]**</mark>\
> <mark>**model**: *Which model to run (i.e., MMETHANE or benchmarker models.) Choices are "mmethane", "ffn" 
> (feed-forward network), "lr" (lasso logistic regresion), "rf" (random forest), or "adaboost" (adaptive gradient boosting)*</mark>
> 
> <mark>**seed**: *Seed (or seeds) to run</mark>
> 
> **run_name**: *(optional): Name of folder in "out_path" for results. If not specified, will use "tag"
> 
> *Additional arguments to "lightning_trainer.py" (if "model"="mmethane"), "lightning_trainer_full_nn.py" (if "model"="ffn") or 
> "benchmarker.py" (if "model"="lr","rf", or "adaboost") can be specified here \
> For example, "dtype" is an input argument to "lightning_trainer.py", which specifies whether to run the model with only metabolite data, only taxonomic data, or both types of data, and can be specified here as:\
> **dtype**: metabs, otus
> 
> 
> <mark>**[sequence_data]**</mark>\
> <mark>**data_type**:*Specify if data is from 16s amplicon sequencing or WGS/metagenomics. Options are* [16s, WGS]</mark>
> 
> <mark>**data**:*sequence data file, as .csv or .tsv. Format expected is rows of samples and columns of sequences. Sequences can
be named with identifiers or with the actual RNA sequence string.*</mark>
> 
> **sequences**:*file with RNA sequences*. <mark>*Required only if data_type==16s AND column labels in 
> sequence_data are not the actual RNA sequence string.</mark> Should be either a .csv/.tsv file with ASV labels 
> (matching column labels in data) as the index and sequences as the first column; or a .fa/.txt file with the format 
> "ASV_label>\nSEQUENCE" for all sequences*
> 
> **taxonomy**:${description:in_path}/*input_taxonomy_file as csv or tsv (i.e., taxonomy file obtained from processing 
> 16S sequences from Dada2)* <mark>*Required only if data_type==16s OR column names in **data** do not contain taxonomy</mark>
> (i.e., If you have metagenomics data processed with Metaphlan, this input is not required)
> 
> **tree**:*(optional) newick_tree. If not specified, pplacer will run to create tree (this may take a bit of time)*
> 
> **distance_matrix**:*(optional) squareform distance matrix as csv or tsv, with rows and column labels corresponding to 
> columns of sequence data. If not specified, will be calculated from phylogenetic distances*
> 
> **replicates**:*(optional) csv or tsv of replicates, for later calculation of expected measurement variance*
>
> **[sequence_preprocessing]** *(optional section specifying data filtering/transformations)*\
> **process_before_training**:*(optional, options=*[True, False]*) whether to process data during training according to 
> training set (thereby preventing data leakage), or beforehand. Note that prediction results may be inflated if set to 
> True. Defaults to False if not specified.*
> 
> **percent_present_in**:*(optional) percent of samples the microbe must be present above limit_of_detection; below this, 
> microbes are filtered out*
> 
> **limit_of_detection**:*(optional) defaults to 0 if not provided*
> 
> **cov_percentile**:*(optional) if specified, get each metabolite's coefficient of variation and keep only metabolites 
> in the top cov_percentile*
> 
> **transformations**:*(optional, options=*[relative_abundance, None]*). Whether to transform sequence data to relative 
> abundance*
>
> <mark>**[metabolite_data]**</mark>\
> <mark>**data**:${description:in_path}/*metabolite data file, as tsv or csv. Samples should be as rows and metabolites as columns.</mark>
> 
> <mark>**meta_data**:${description:in_path}/*metabolite meta-data file, as tsv or csv. Indices should correspond to 
> metabolite feature names in metabolitee data. Data frame columns should include either HMBD ids (labeled 'HMDB' or 
> 'hmdb'), KEGG ids (labeled 'KEGG' or 'kegg') or inchikeys (labeled 'InChIKey' or 'inchikey')*</mark>
> 
> **taxonomy**:*(optional) Path to classy fire classifications for this set of metabolites. If not available, 
> data processor will get these from the metabolite meta data*
> 
> **fingerprint_type**:*(optional) which type of fingerprint to get for each metabolite, defaults to pubchem. 
> Options are: pubchem, rdkit, morgan, mqn. Not needed if similarity/distance matrix is supplied.*
> 
> **similarity_matrix**:*(optional) path to similarity matrix. Will be calculated if not specified.*
> 
> **distance_matrix**:*(optional) path to distance matrix. Will be calculated if not specified.*
>
> **[metabolite_preprocessing]** *(Optional section specifying metabolite preprocessing)*\
> **process_before_training**:*(optional, options=*[True, False]*) whether to process data during training according 
> to training set (thereby preventing data leakage), or beforehand. Defaults to False if not specified*
> 
> **percent_present_in**:*(optional) percent of samples the metabolite must be present above "limit_of_detection" to not be filtered out*
> 
> **limit_of_detection**:*(optional) defaults to 0 if not specified*
> 
> **cov_percentile**:*(optional) if specified, get each metabolite's coefficient of variation and keep only metabolites 
> in the top cov_percentile*
> 
> **transformations**:*(optional, options=*[log, standardize] *or both (i.e. log, standardize)*