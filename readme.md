## Package dependencies
- python 3.6
- ete3 (v3.1.2)
- ete_toolchain 
- matplotlib (v3.3.4)
- bitarray (v2.3.0)
- hmmer (v3.3.2)
- guppy3 (v3.1.0)
- rdkit
- statsmodels (v0.12.2)
- scipy (v1.5.4)
- scikit-learn (v0.24.2)


## Process data from a config file

Create a config file for your dataset. **config_files/config_sample.cfg** is a sample config file with all options outlined. Config file options are the following:
> **[description]** \
> **tag**:*name of your dataset*\
> **in_path**:*(optional) path of input data folder*\
> **out_path**:*(optional) path to save processed data; defaults to current working directory if not specified*\
>
> **[data]**\
> **subject_data**:${description:in_path}*subject meta-data file, as .csv or .tsv. Data should have samples as the rows and the outcome as a column. If data is to be split on a coviariate, the covariate should also be included as a column.*\
> **outcome_variable**:*Column name in subject_data that has subject outcomes*\
> **outcome_positive_value**:*(optional) If outcome values are not ints (0s and 1s), specify the outcome string that signifies the outcome has occured*\
> **outcome_negative_value**:*(optional) Can specify instead of outcome_positive_value or can specify both. Specifying both is useful if data has some samples with a third outcome; these samples won't be included in processed data*\
> **sample_id_column**:*(optional) Use to signify the sample IDs column in subject_data if the sample IDs are not the index of subject_data*\
> **covariate_variable**:*(optional) If needed, specify covariate variable that data should be split by (i.e. race for vaginal microbiome data)*\
> **covariate_positive_value**:*(optional) If specifying covariate_variable and covariate values are not ints (0s,1s) specify covariate positivd value*\
> **covariate_negative_value**:*(optional) Can specify instead of outcome_positive_value or can specify both (i.e. if some samples have third covariate that shouldn't be in processed data)*\
>
> **[sequence_data]**\
> **data_type**:*Specify if data is from 16s amplicon sequencing or WGS/metagenomics. Options are* [16s, WGS]\
> **data**:${description:in_path}/*sequence data file, as .csv or .tsv. Format expected is rows of samples and columns of sequences. Sequences can
be named with identifiers or with the actual RNA sequence string.*\
> **sequences**:#${description:in_path}/*file with RNA sequences*. *Required only if column labels in sequence_data are not the actual RNA sequence string. Should be either a .csv/.tsv file with ASV labels (matching column labels in data) as the index and sequences as the first column; or a .fa/.txt file with the format "ASV_label>\nSEQUENCE" for all sequences*\
> **taxonomy**:${description:in_path}/*input_taxonomy_file as csv or tsv*\ 
> **tree**:*(optional) newick_tree. If not specified, pplacer will run to create tree (this may take a bit of time)*\
> **distance_matrix**:*(optional) squareform distance matrix as csv or tsv, with rows and column labels corresponding to columns of sequence data. If not specified, will be calculated from phylogenetic distances*\
> **replicates**:*(optional) csv or tsv of replicates, for later calculation of expected measurement variance*\
>
> **[sequence_preprocessing]** *(optional section specifying data filtering/transformations)*\
> **process_before_training**:*(optional, options=*[True, False]*) whether to process data during training according to training set (thereby preventing data leakage), or beforehand. Note that prediction results may be inflated if set to True. Defaults to False if not specified.*\
> **percent_present_in**:*(optional) percent of samples the microbe must be present above limit_of_detection; below this, microbes are filtered out*\
> **limit_of_detection**:*(optional) defaults to 0 if not provided*\
> **cov_percentile**:*(optional) if specified, get each metabolite's coefficient of variation and keep only metabolites in the top cov_percentile*
> **transformations**:*(optional, options=*[relative_abundance, None]*). Whether to ransform sequence data to relative abundance*\
>
> **[metabolite_data]**\
> **data**:${description:in_path}/*metabolite data file, as tsv or csv. Samples should be as rows and metabolites as columns.*\
> **meta_data**:${description:in_path}/*metabolite meta-data file, as tsv or csv. Indices should correspond to metabolite feature names in metabolitee data. Data frame columns should include either HMBD ids (labeled 'HMDB' or 'hmdb'), KEGG ids (labeled 'KEGG' or 'kegg') or inchikeys (labeled 'InChIKey' or 'inchikey')*\
> **taxonomy**:*(optional) Path to classy fire classifications for this set of metabolites. If not available, data processor will get these from the metabolite meta data*\
> **fingerprint_type**:*(optional) which type of fingerprint to get for each metabolite, defaults to pubchem. Options are: pubchem, rdkit, morgan, mqn. Not needed if similarity/distance matrix is supplied.*\
> **similarity_matrix**:*(optional) path to similarity matrix*\
> **distance_matrix**:*(optional) path to distance matrix*\
> **replicates**:*(optional) path to replicates data as csv or tsv*\
>
> **[metabolite_preprocessing]** *(Optional section specifying metabolite preprocessing)*\
> **process_before_training**:*(optional, options=*[True, False]*) whether to process data during training according to training set (thereby preventing data leakage), or beforehand. Defaults to False if not specified*\
> **percent_present_in**:*(optional) percent of samples the metabolite must be present above "limit_of_detection" to not be filtered out*\
> **limit_of_detection**:*(optional) defaults to 0 if not specified*\
> **cov_percentile**:*(optional) if specified, get each metabolite's coefficient of variation and keep only metabolites in the top cov_percentile*\
> **transformations**:*(optional, options=*[log, standardize] *or both (i.e. log, standardize)*

After specifying config file, run the following command:
> python ./process_data.py --config_file <path/to/your_config_file.cfg>

This creates pickle files in the specified output folder for the metabolite data and the sequence data. Note that if 'coviariate_variable' is specified in the config file, there will be a pickle file for each covariate that the dataset is split by. If not, there is just one file for the metabolites and one for the sequences.

Both pickle files are a dictionary with the following keys:
- **'X'**: the data matrix with samples as the rows and features as the columns
- **'y'**: the binary outcome labels (with 1 as the outcome positive value)
- **'distances'**: the distances (either phylogenetic or fingerprint based) between features (note: for the metabolite data, this matrix will be smaller than the number of features as some metabolites don't have associated fingerprints/IDs)
- **'variable_tree'**: the taxonommic tree, either based on phylogeny or classy-fire classifications
- **'preprocessing'**: if the data has not yet been filtered/transformed (so that the data can be filtered/transformed based on the train split and prevent data leakage), this passes along the filtering/transformation parameters to be used in the model. Otherwise, if data is filtered during pre-processing (done by setting 'process_before_training' to True in the config file), this is equal to **None**
- **'taxonomy'**: taxonomy data frame, where the columns correspond to the feature names in 'X' and the rows are, for metabolites, the classy-fire classifications for ('superclass','class','subclass','level 5',...'level 10') or, for microbes, the taxonomic levels ('Kingdom','Phylum','Class','Order','Family','Genus','Species')
- **'replicates'**: if path to replicates csv/tsv is provided in config file, this key contains the replicate data
The sequence data dictionary also has a key **'sequences'**, which contains a data-frame with the microbial sequences

## Utility functions

These functions are located in **utilities/util.py**


### Calculate radii prior from taxonomic information

Function **outputs=calculate_radii_prior(inputs)** calculates the mean and variance of the prior distribution for metabolite/microbial radii. Mean is calculated as the median of medians of family/subclass level distances, and variance is the variance of the medians of family/subclass level distances. \

Inputs:
- dataset_dict: output dataset dictionary from process_data.py (must contain keys "taxonomy" and "distances")
- dtype: whether the data is metabolomics or sequences (options=['metabs','otus'])
- multiplier: multiplier for variance, defaults to 1

Outputs:
- dictionary with keys "mean" and "variance", for the mean and variance of the truncated normal prior (or the exponentiated mean and variance for a lognormal prior)


### Split dataset into train and test splits, and filter and transform based on training set

Function **split_and_preprocess_dataset()** splits dataset into train and test splits given and filters/transforms based on the training set IF that filtering was not done in data preprocessing.

Inputs:
- dataset_dict: output dataset dictionary from process_data.py
- train_ixs: list of index locations for train set
- test_ixs: list of index locations for test set

Outputs:
- train_dataset_dict: dict with all the same keys as dataset dict, but with just the training subjects and the features after filtering
- test_dataset_dict: dictionary with test subjects and filtered/transformed based on test set
   
 
### Initialize detector / cluster locations and radii with KNN

Function **init_w_knn()**

Inputs:
- emb_locs: [num_features x embedding_dimension] array of embedded feature 'locations'
- dist_mat: [num_features x num_features] array of distances between emb_locs
- size: list with size of locations/radii to initialize (i.e. for MDITRE, input is [num_rules, num_detectors]. For M2M, input will be [num_clusters])
- seed: random seed

Outputs:
- kappa_init: initial radii for all rules and detectors
- eta_init: initial centers for all rules and detectors
- detector ids: list of lists of which features are assigned to which cluster


### Initialize unknown embedding locations with gaussian mixtures (GMs)

Function **init_w_gms()** initializes embedding locations for unknown metabolites, when we learn the embedded locations from model

Inputs:
- N_feat: # of unknown features to initialize location of
- N_clust: # of mixtures (for MDITRE, set to num_detectors)
- dim: embedding dimension
- r_desired: either a dictionary of the mean and variance to generate a N_clust radii from a log-normal distribution 
  for the gaussian mixtures, or a list of N_clust radii for the gaussian mixtures
  - Can use input dictionary from **calculate_radii_prior()**
- mu_met: array of [N_clust x emb_dim] centers around which to generate the GMs, or None
    - if None, centers are generated from multivariate normal centered at 0 with variance 0.1
- r_mult: factor to multiply the radius by (added because the gaussian mixture model with input from calculated radii prior was too expansive); 
    - default = 0.01

Outputs:
- met_locs: generated embedded initial locations
- mu_met: generated (or input) GM centers
- r_met: generated (or input) GM radii
- ids: which features belong to which intial gaussian mixture


### Compute pairwise distances given feature matrix

Function **compute_dist()** is used mostly for computing the embedded distance matrix after embedding

Inputs:
- dist_emb: [num_features x embedding_dimension] array of embedded locations
- num_otus: num_features

Outputs:
- dist: [num_features x num_features] distance matrix array


### Embed distance matrix to d dimensions

Function **compute_dist_emb_mds** embeds a distance matrix using SVD

Inputs:
- dist_matrix: [num_features x num_features] distance matrix array (pre-embedding)
- emb_dim: dimension to embed
- seed: random seed

Outputs:
- [num_features x embedding_dimension] array of embedded locations


### Choose dimension for embedding

Function **test_d_dimensions()** finds the lowest dimension that results in a Kolmogorov-Smirnov p-value greater than 0.05.
If no dimension tested exceeds a 0.05 p-value, returns the dimension with the greatest p-value

Inputs: 
- d: list of dimensions to test 
- dist_matrix: [num_features x num_features] distance matrix array (pre-embedding)
- seed: random seed

Outputs:
- emb_dim: chosen embedding dimension
- dist_emb: [num_features x embedding_dimension] array of embedded locations at chosen dimension 
- dist_matrix_emb: [num_features x num_features] array of distances b/w embeddings


### Ensure that both datasets have the same samples in the data X and labels y

Function **merge_datasets()**

Inputs:
- dataset_dict: dictionary of dataset-dictionaries generated from process_data.py, where each key is the data type (i.e. {'metabs': metabolite-data-dictionar, 'otus': sequence-data-dictionary)

Outputs:
- dataset_dict: same dictionary, but with ensuring that indices in X and y for both datasets match
- y: [N_subjects] outcome labels (i.e. 'y' in either dataset dictionary)


### Split data by k-folds

Function **cv_kfold_splits()**

Inputs:
- X: [N_subjects x N_features] data array, OR [N_subjects] array of zeros
- y: [N_subjects] array of outcomes
- num_splits: number of k-folds (defaults to 5)
- seed: random seed

Outputs:
- train_ids: list of k lists, where each list is the train index location for that fold
- test_ids: list of k lists, where each list is the test index location for that fold


### Split data by LOO

Function **cv_loo_splits()**

Inputs:
- X: [N_subjects x N_features] data array, OR [N_subjects] array of zeros
- y: [N_subjects] array of outcomes

Outputs:
- train_ids: list of k lists, where each list is the train index location for that fold
- test_ids: list of k lists, where each list is the test index location for that fold
