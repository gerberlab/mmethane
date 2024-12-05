import pickle, warnings, json, itertools, re, copy

from scipy.spatial.distance import squareform, pdist

from .data_utils import *
import configparser

import numpy as np
import pandas as pd
try:
    import ete4
except:
    try:
        import ete3 as ete4
    except:
        pass


import os
import pickle as pkl

class ProcessData:
    def __init__(self, config_filename):
        # Open config file
        if isinstance(config_filename, str):
            self.config = configparser.RawConfigParser(interpolation=configparser.ExtendedInterpolation())
            self.config.read(config_filename)
        else:
            self.config = config_filename
        self.fingerprint_options=['pubchem','infomax','map4','rdkit','morgan','mqn','mhfp','mxfp']

        if 'metabolite_data' in self.config and 'fingerprint_type' in self.config['metabolite_data'] and 'week' not in self.config['description']['tag']:
            if self.config['metabolite_data']['fingerprint_type'].lower() not in self.config['description']['tag'].lower():
                if self.config['description']['tag'].split('_')[-1].lower() in self.fingerprint_options:
                    raise Warning('different fingerprint name in tag than fingerprint specified in "fingerprint_type". Make sure tag/fingerprint_type is correct!')


        # Create folder to save datasets in. Folder will be in 'out_path' with the name <tag>_<fingerprint>_<week>
        self.save_path=self.config['description']['out_path'] + self.config['description']['tag']
        if not os.path.isdir(self.config['description']['out_path']):
            os.mkdir(self.config['description']['out_path'])

        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        # save copy of config file in new dataset directory so that you can always know what config file you used to
        # generate the data in the folder
        with open(self.save_path + '/config_ran.cfg', 'w') as f:
            self.config.write(f)

        if 'input_path' in self.config['data'] and self.config['data']['input_path'] is not None:
            self.in_path = self.config['data']['input_path']
        else:
            self.in_path = os.getcwd()

        # load subject data
        self.Y, self.subject_data, self.subject_IDs = self.load_subject_data(self.config['data'])

        # If additional_subject_covariates, check to make sure covariates are in subject data
        if 'additional_subject_covariates' in self.config['data'] and self.config['data']['additional_subject_covariates']!='':
            self.check_additional_subj_covariates(self.config['data']['additional_subject_covariates'])


        # If sequence data file in config, load sequence data
        if 'sequence_data' in self.config and self.config['sequence_data']['data'] is not None:
            self.data_type = self.config['sequence_data']['data_type'].lower()

            # Load data based on type of data (16s vs WGS) and raise an error if data_type is not 16s or WGS/wgs
            if self.data_type=='16s':
                self.sequence_data, self.sequences = self.load_16s_data(self.config['sequence_data']['data'])
            elif self.data_type=='wgs':
                self.sequence_data, self.sequences = self.load_wgs_data(self.config['sequence_data']['data'])
                self.metaphlan = True
                # if all([len(s.split(';')) >=6 for s in self.sequence_data.columns.values]):
                #     self.metaphlan = True
            else:
                raise ValueError('Please provide valid data type for sequence data. Options are "16s" and "WGS".')

            # Sequence data filtering & transformations
            self.sequence_preprocessing = None
            if 'sequence_preprocessing' in self.config:
                # filter sequence data if process_before_training == True; otherwise, keep track of filtering parameters
                # to add to data dictionary so that filtering can be done during training
                if 'process_before_training' in self.config['sequence_preprocessing'] and self.config['sequence_preprocessing']['process_before_training']=='True':
                    self.sequence_data = self.filter_data(self.sequence_data, self.config['sequence_preprocessing'], 'sequences')
                    # self.sequence_preprocessing=None
                    if 'transformations' in self.config['sequence_preprocessing'] and \
                            self.config['sequence_preprocessing']['transformations'] is not None:
                        self.transform_sequence_data()
                else:
                    # Filter out OTUs present in 1 or fewer participants so that getting phylogenetic distances doesn't
                    # take so long (these would be filtered out during training regardless of test/train split)
                    ppi_conservative = np.ceil((1/(self.subject_data.shape[0]))*100)
                    if 'limit_of_detection' in self.config['sequence_preprocessing'] and self.config['sequence_preprocessing']['limit_of_detection']!='':
                        lod=float(self.config['sequence_preprocessing']['limit_of_detection'])
                    else:
                        lod=0
                    orig_num=self.sequence_data.shape[1]
                    self.sequence_data = filter_by_presence_func(self.sequence_data,
                                                                 perc_present_in=ppi_conservative, limit_of_detection=lod)
                    self.sequence_preprocessing=self.get_filtering_parameters(self.config['sequence_preprocessing'])
                    print(f'Conservatively filtered microbes from {orig_num} to {self.sequence_data.shape[1]} present in '
                          f'less than 1% of the {self.sequence_data.shape[0]} participants '
                          f'to save time finding phylogenetic distances.')

                    # if 'transformations' in self.config['sequence_preprocessing'] and self.config['sequence_preprocessing']['transformations'] is not None:

            if self.data_type=='wgs':
                self.taxa_strings = self.sequence_data.columns.values
                self.seq_label_dict = get_asv_labels(self.sequence_data.columns.values)
                self.sequence_data = self.sequence_data.rename(columns=self.seq_label_dict)

            if 'tree' in self.config['sequence_data'] and self.config['sequence_data']['tree'] != '':
                # If tree is specified in sequence file, just load tree
                self.sequence_tree = ete4.Tree(self.config['sequence_data']['tree'])
            elif self.data_type=='wgs':
                print('\nMaking WGS tree')
                # Make WGS sequence tree (need to update this to make real tree!!)
                if self.metaphlan==True:
                    if 'reference_tree' not in self.config['sequence_data'] or self.config['sequence_data']['reference_tree'] =='':
                        ImportError('No reference tree file found. Please provide a reference tree file if using WGS')
                    if 'reference_mapper' not in self.config['sequence_data'] or self.config['sequence_data']['reference_mapper'] =='':
                        reference_mapper=None
                    else:
                        reference_mapper=self.config['sequence_data']['reference_mapper']
                    tree, self.reference_dict = make_wgs_tree(self.taxa_strings,self.config['sequence_data']['reference_tree'],
                                         reference_mapper)
                    self.sequence_tree = relabel_tree(tree, self.seq_label_dict)
                    with open(self.save_path + '/sequence_tree.nhx', 'w') as f:
                        f.write(self.sequence_tree.write(format=0))
                else:
                    raise KeyError('If WGS data is not from metaphlan (and formatted with taxa strings as column labels), a pre-made tree must be provided')

            elif self.data_type=='16s':
                # else, if data type is 16s and 16s sequences are provided (either in config file or as column names in sequence data), make 16s tree
                if 'sequences' in self.config['sequence_data'] and self.config['sequence_data']['sequences']!='' and self.sequences is not None:
                    self.sequence_tree=make_16s_tree(self.config['sequence_data']['sequences'], self.sequences, self.sequence_data, self.save_path)
                else:
                    # sequence_names = [f'ASV {i}' for i in range(self.sequences.shape[1])]
                    # seq_df = pd.Series(self.sequences.columns.values, index = sequence_names)
                    self.sequences.to_csv(os.path.join(self.save_path,'sequences.csv'))
                    self.sequence_tree = make_16s_tree(os.path.join(self.save_path,'sequences.csv'), self.sequences,
                                                       self.sequence_data, self.save_path)
                    # Otherwise, ask user to provide sequences
                    # raise KeyError(
                    #     'No sequence or tree file was provided. A file mapping bacteria names to sequences or a tree '
                    #     'file in newick format is required for '
                    #     '16s data. Please provide "sequences="<filename> or "tree"=<filename> under section "sequence data"')

            # If distance matrix isn't provided, get distances from tree
            if 'distance_matrix' not in self.config['sequence_data'] or self.config['sequence_data']['distance_matrix']=='':
                self.sequence_dist_df = get_sequence_distance_matrix(self.sequence_data, self.sequence_tree)
                self.sequence_dist_df.to_csv(self.save_path+'/seq_dist.csv')
            # Otherwise load distance matrix
            else:
                self.sequence_dist_df = pd.read_csv(self.config['sequence_data']['distance_matrix'], index_col=0)

            print(f'{self.sequence_dist_df.shape[0]} taxa have associated distances')
            if 'keep_sequences_without_distances' in self.config['sequence_data'] and self.config['sequence_data']['keep_sequences_without_distances'].lower()=='true':
                print(f'{self.sequence_data.shape[1] - self.sequence_dist_df.shape[0]} taxa without associated distances will not be removed from sequence dataset')
            else:
                self.sequence_data = self.sequence_data[self.sequence_dist_df.index.values]
            # If taxonomy provided,
            self.taxonomy=None
            if 'taxonomy' in self.config['sequence_data'] and self.config['sequence_data']['taxonomy']!='':
                self.taxonomy=self.format_16s_taxonomy(self.config['sequence_data']['taxonomy'])
            elif self.metaphlan==True:
                self.taxonomy=get_taxa_df_from_strings(self.taxa_strings, self.seq_label_dict, self.reference_dict)

            self.save_sequence_data_to_pickle()


        if 'metabolite_data' in self.config and self.config['metabolite_data']['data']!='':
            # LOAD METABOLITE DATA
            self.load_metabolite_data()
            self.load_metabolite_meta_data()
            cols_no_cap = [s.lower() for s in self.metabolite_meta_df.columns.values]
            self.metabolite_meta_df.columns = cols_no_cap
            # if 'collapse' in self.config['metabolite_data'] and self.config['metabolite_data']['collapse'].replace(' ','').lower()=='true':
            #     self.metabolite_data = collapse_metabolites(self.metabolite_data, self.metabolite_meta_df)
            self.metabolite_preprocessing = None
            self.collapse=True
            if 'metabolite_preprocessing' in self.config:
                if 'process_before_training' in self.config['metabolite_preprocessing'] and self.config['metabolite_preprocessing']['process_before_training'] == 'True':
                    # self.metabolite_preprocessing = None
                    if 'collapse' in self.config['metabolite_data'] and self.config['metabolite_data'][
                        'collapse'].replace(' ', '').lower() == 'true':

                        self.metabolite_data, self.metabolite_meta_df = collapse_metabolites(self.metabolite_data, self.metabolite_meta_df)
                        self.collapse = False
                    self.metabolite_data = self.filter_data(self.metabolite_data, self.config['metabolite_preprocessing'], 'metabolites')
                    self.transform_metabolite_data()
                else:
                    self.metabolite_preprocessing=self.get_filtering_parameters(self.config['metabolite_preprocessing'])
                    if 'transformations' in self.config['metabolite_preprocessing'] and self.config['metabolite_preprocessing']['transformations']!='':
                        self.metabolite_preprocessing['transformations'] = self.config['metabolite_preprocessing']['transformations']


            # self.load_metabolite_meta_data()
            # cols_no_cap = [s.lower() for s in self.metabolite_meta_df.columns.values]
            # self.metabolite_meta_df.columns = cols_no_cap
            self.fingerprint=None
            if 'distance_matrix' in self.config['metabolite_data'] and self.config['metabolite_data']['distance_matrix']!='':
                self.metabolite_dist_df = read_ctsv(self.config['metabolite_data']['distance_matrix'])
            elif 'similarity_matrix' not in self.config['metabolite_data'] and self.config['metabolite_data']['similarity_matrix']!='':
                self.metabolite_dist_df = 1-read_ctsv(self.config['metabolite_data']['similarity_matrix'])
            else:
                if 'fingerprint_type' in self.config['metabolite_data'] and self.config['metabolite_data'][
                    'fingerprint_type']!='':
                    self.fingerprint = self.config['metabolite_data']['fingerprint_type']
                else:
                    self.fingerprint = 'pubchem'

                # define which intermediary is needed to go from HMDB/KEGG to fingerprint
                if self.fingerprint.lower() == 'pubchem':
                    self.key_type = 'fingerprints'
                elif self.fingerprint.lower() == 'rdkit' or self.fingerprint.lower() == 'morgan' \
                        or self.fingerprint.lower()=='mqn' or self.fingerprint.lower()=='infomax':
                    self.key_type = 'InChI Code'
                elif self.fingerprint.lower() == 'mhfp' or self.fingerprint.lower() == 'mxfp' or self.fingerprint.lower()=='map4':
                    self.key_type = 'SMILES'

                # if fingerprints are in meta data, just get distance matrix
                if 'fingerprints' in cols_no_cap:
                    self.get_metabolite_distance_matrix()

                # otherwise, if key type in distance matrix, get fingerprints and then distance matrix
                elif self.key_type.lower() in cols_no_cap:
                    self.get_metabolite_fingerprints()
                    self.get_metabolite_distance_matrix()

                # else, get intermediary IDs (self.key_type) from HMDB/KEGG ids, then get fingerprints, and then get distance matrix
                elif 'hmdb' in cols_no_cap or 'kegg' in cols_no_cap or 'inchikey' in cols_no_cap:
                    self.get_metabolite_intermediate_ids(key_type=self.key_type)
                    self.get_metabolite_fingerprints()
                    self.get_metabolite_distance_matrix()
                else:
                    raise ValueError('Cannot find metabolite fingerprints or IDs in metabolite metadata. '
                                     'Must provide either "fingerprints", "inchikey", "hmdb", and/or "kegg" ids as '
                                     'column in metabolite metadata file')

            if 'taxonomy' not in self.config['metabolite_data'] or self.config['metabolite_data']['taxonomy'] is None:
                if 'skip_taxonomy' in self.config['metabolite_data'] and self.config['metabolite_data']['skip_taxonomy'].lower()=='true':
                    self.metabolite_classifications = pd.DataFrame(columns = self.metabolite_data.columns.values,
                                                                   index=['kingdom', 'superclass', 'class', 'subclass',
                                                                          'level 5', 'level 6','level 7', 'level 8',
                                                                          'level 9', 'level 10'])
                else:
                    if self.key_type!='InChIKey' and 'inchikey' not in self.metabolite_meta_df.columns.values:
                        self.get_metabolite_intermediate_ids(key_type='InChIKey')
                    self.get_metabolite_classifications()
            else:
                self.metabolite_classifications = pd.read_csv(self.config['metabolite_data']['taxonomy'], header=0,
                                                              index_col=0)

            if 'tree' not in self.config['metabolite_data'] or self.config['metabolite_data']['tree'] is None:
                if 'skip_taxonomy' in self.config['metabolite_data'] and self.config['metabolite_data']['skip_taxonomy'].lower()=='true':
                    self.metabolite_tree = ete4.Tree()
                else:
                    self.get_metabolite_tree()
            else:
                self.metabolite_tree = ete4.Tree(self.config['metabolite_data']['tree'])

            self.save_metabolite_data_to_pickle()

    def load_subject_data(self, meta_data):
        print('\nLoading subject data')
        subject_file = meta_data['subject_data']
        outcome_variable = meta_data['outcome_variable']
        if 'outcome_positive_value' in meta_data and meta_data['outcome_positive_value']!='':
            outcome_positive_value = meta_data['outcome_positive_value']
            try:
                outcome_positive_value = int(outcome_positive_value)
            except:
                pass
        else:
            outcome_positive_value=1
        subject_data = read_ctsv(subject_file)
        N_o = subject_data.shape[0]
        if 'sample_id_column' in meta_data and meta_data['sample_id_column'] != '':
            subject_data = subject_data.set_index(
                meta_data['sample_id_column'])

        #
        if isinstance(outcome_positive_value, str):
            if ',' in outcome_positive_value:
                outcome_positive_value = sum([s.split(', ') for s in outcome_positive_value.split(',')], [])
        if 'outcome_negative_value' in meta_data:
            outcome_negative_value = meta_data['outcome_negative_value']
            try:
                outcome_negative_value = int(outcome_negative_value)
            except:
                pass
            if isinstance(outcome_negative_value, str):
                if ',' in outcome_negative_value:
                    outcome_negative_value = sum([s.split(', ') for s in outcome_negative_value.split(',')], [])
        else:
            outcome_negative_value = list(set(subject_data[outcome_variable].values)-set([outcome_positive_value]))

        if (isinstance(outcome_positive_value, str) or isinstance(outcome_positive_value, int)) and 'outcome_negative_value' not in meta_data:
            Y = (subject_data[outcome_variable] == outcome_positive_value).astype(int)
            # outcome_negative_value = list(set(subject_data[outcome_variable].values)-set([outcome_positive_value]))[0]
        else:
            if isinstance(outcome_positive_value, str):
                outcome_positive_value = [outcome_positive_value]
            y_ls = {}
            for s_id, value in subject_data[outcome_variable].items():
                if value in outcome_positive_value:
                    y_ls[s_id] = 1
                elif value in outcome_negative_value:
                    y_ls[s_id] = 0
            Y = pd.Series(y_ls)

        if isinstance(outcome_positive_value, list):
            if len(outcome_positive_value)>1:
                outcome_positive_value = ','.join(outcome_positive_value)
            else:
                outcome_positive_value=outcome_positive_value[0]

        if isinstance(outcome_negative_value, list):
            if len(outcome_negative_value)>1:
                outcome_negative_value = ','.join(outcome_negative_value)
            else:
                outcome_negative_value=outcome_negative_value[0]

        self.class_labels={1:outcome_positive_value,0:outcome_negative_value}
        if sum(Y)>(len(Y)-sum(Y)):
            print('Warning: It looks like the positive class is the more abundant class. \n'
                  'This might cause falsely high F1 scores if the predictor predicts the more abundant class.\n'
                  'Try setting "outcome_positive_value" to the less abundant class.')
        if any(Y.isna()):
            ixs = Y.isna()
            subject_data = subject_data.loc[~ixs]
            Y = Y[~ixs]
        subject_data = subject_data.loc[Y.index.values]
        subject_IDs = subject_data.index.values

        print(f'Subject meta data has {N_o} original subjects\n'
              f'{np.sum(Y.values)} subjects have {outcome_variable}={outcome_positive_value},\n'
              f'{np.sum(Y.values == 0)} subjects have {outcome_variable}={outcome_negative_value}\n'
              f'Combined, this leaves {Y.shape[0]} subjects in the dataset to be analyzed.')
        return Y, subject_data, subject_IDs

    def check_additional_subj_covariates(self,addn_subj_covariates):
        # self.addn_sub_covars = self.config['data']['additional_subject_covariates']
        for covar in addn_subj_covariates:
            if covar not in self.subject_data.columns.values:
                raise KeyError('Trying to control for covariate %s, but no "%s" '
                               'column in subject_data.' % (covar, covar))
            elif len(np.unique(self.subject_data[covar]))==1:
                raise KeyError('Trying to control for covariate %s, but ')

    def filter_data(self, data, preprocessing_section, data_type):
        print('\nFiltering metabolite data')
        if 'limit_of_detection' in preprocessing_section and preprocessing_section['limit_of_detection']!='':
            lod = float(preprocessing_section['limit_of_detection'])
        else:
            lod=0

        if 'percent_present_in' in preprocessing_section and preprocessing_section['percent_present_in']!='':
            ppi = float(preprocessing_section['percent_present_in'])
            data = filter_by_presence_func(data, perc_present_in=ppi,
                                                           limit_of_detection=lod)
            print('After filtering {3} to keep only {3} with levels > {0} in {1}\% of participants, '
                  '{2} remain'.format(lod, ppi,data.shape[1], data_type))

        if 'cov_percentile' in preprocessing_section and preprocessing_section['cov_percentile']!='':
            cp = int(preprocessing_section['cov_percentile'])
            data=filter_by_cov_func(data, cov_percentile=cp)
            print('After filtering {2} to keep only {2} with coefficients of variation in the '
                  'top {0} percentile of participants, {1} {2} remain'.format(
                cp, data.shape[1], data_type))
        return data


    def load_16s_data(self, sequence_file):
        print('\nLoading 16s data')
        sequence_data = read_ctsv(sequence_file)
        if 'samples_dimension' in self.config['sequence_data'] and self.config['sequence_data']['samples_dimension']=='columns':
            sequence_data = sequence_data.T
        sub_ids = [s for s in self.subject_IDs if s in sequence_data.index.values]
        sequence_data = sequence_data.loc[sub_ids]
        sequence_data=sequence_data[~sequence_data.index.duplicated(keep='first')]
        if all([set(c) == {'A', 'C', 'G', 'T'} for c in sequence_data.columns.values]):
            lab_to_seq = {f'ASV_{i}': v for i, v in enumerate(sequence_data.columns.values)}
            sequences = pd.Series(lab_to_seq)
            seq_to_lab = {v: k for k, v in lab_to_seq.items()}
            sequence_data = sequence_data.rename(columns=seq_to_lab)
        else:
            sequences = None
        print('Loaded 16s data has {0} samples and {1} genera'.format(
            sequence_data.shape[0], sequence_data.shape[1]))
        return sequence_data, sequences

    def load_wgs_data(self, seq_data):
        print('\nLoading WGS data')
        sequences=None
        sequence_data = read_ctsv(seq_data)
        if 'samples_dimension' in self.config['sequence_data'] and self.config['sequence_data']['samples_dimension']=='columns':
            sequence_data = sequence_data.T
        keep_ixs = [s for s in self.subject_IDs if s in sequence_data.index.values]
        sequence_data = sequence_data.loc[keep_ixs]
        print('Loaded WGS data has {0} samples and {1} genera'.format(
            sequence_data.shape[0], sequence_data.shape[1]))
        return sequence_data, sequences

    def transform_sequence_data(self):
        if 'relative_abundance' in self.config['sequence_preprocessing']['transformations']:
            print('\nTransforming sequence data')
            self.sequence_data = self.sequence_data.divide(self.sequence_data.sum(1), axis='index')
            mn = np.mean(self.sequence_data.values,0).mean()
            std = np.std(self.sequence_data.values,0).mean()
            print('After transforming to relative abundance, ASVs have an average mean of {0} '
                  'and an average standard deviation of {1}'.format(mn, std))



    def format_16s_taxonomy(self,taxonomy_file):
        taxa_labels = pd.read_csv(taxonomy_file, index_col=0)
        if taxa_labels.shape[0] > taxa_labels.shape[1]:
            taxa_labels = taxa_labels.T
        if all([set(c) == {'A', 'C', 'G', 'T'} for c in taxa_labels.columns.values]) and self.sequences is not None:
            seq_to_lab = {v: k for k, v in self.sequences.to_dict().items()}
            taxa_labels = taxa_labels.rename(columns=seq_to_lab)

        dual = list(set(taxa_labels.columns.values).intersection(self.sequence_data.columns.values))
        if len(dual) == 0:
            raise ValueError("No sequences have taxonomy information. Double check that labels correspond.")
        elif len(dual) < self.sequence_data.shape[1]:
            print(f"WARNING: only {len(dual)}/{self.sequence_data.shape[1]} ASVs have taxonomic information.")
        taxa_labels = taxa_labels[dual]
        return taxa_labels


    def load_metabolite_data(self):
        print('\nLoading metabolite data')
        self.metabolite_data = read_ctsv(self.config['metabolite_data']['data'])
        if 'samples_dimension' in self.config['metabolite_data'] and self.config['metabolite_data']['samples_dimension']=='columns':
            self.metabolite_data = self.metabolite_data.T
        sub_ids = [s for s in self.subject_IDs if s in self.metabolite_data.index.values]
        self.metabolite_data = self.metabolite_data.loc[sub_ids]
        self.metabolite_data = self.metabolite_data[~self.metabolite_data.index.duplicated(keep='first')]
        # self.metabolite_data = self.metabolite_data.loc[self.subject_IDs]
        print('Loaded metabolite data has {0} samples and {1} metabolites.\n'
              'The data has an average mean of {2} and an average standard deviation of {3} across metabolites,'
              'and ranges from {4} to {5}'.format(
            self.metabolite_data.shape[0], self.metabolite_data.shape[1], np.mean(self.metabolite_data.values, 0).mean(),
            np.std(self.metabolite_data.values, 0).mean(),self.metabolite_data.min().min(),self.metabolite_data.max().max()))

    def transform_metabolite_data(self):
        print('\nTransforming metabolite data')
        if 'transformations' in self.config['metabolite_preprocessing']:
            if 'log' in self.config['metabolite_preprocessing']['transformations'] and 'standardize' in self.config['metabolite_preprocessing']['transformations']:
                self.metabolite_data = transform_func(self.metabolite_data)
                print('After standardizing data, metabolites have an average mean of {0} '
                      'and an average standard deviation of {1}, and ranges from {2} to {3}'.format(
                    np.mean(self.metabolite_data.values,0).mean(), np.std(self.metabolite_data.values,0).mean(),
                    self.metabolite_data.min().min(),self.metabolite_data.max().max()))


    def load_metabolite_meta_data(self):
        print('\nLoading metabolite meta data')
        if 'meta_data' in self.config['metabolite_data']:
            try:
                self.metabolite_meta_df = read_ctsv(self.config['metabolite_data']['meta_data'], index_col=0, header=0)
            except KeyError:
                raise KeyError('No meta data file for metabolites in config file. Please include '
                               '"meta_data=<path/to/metabolite_meta_data/>" in config file under section "metabolite_data"')

            self.metabolite_meta_df.columns = [m.lower() for m in self.metabolite_meta_df.columns.values]

            missing_meta_data = list(
                set(self.metabolite_data.columns.values) - set(self.metabolite_meta_df.index.values))
            if len(missing_meta_data) > 0:
                print('Warning: {0} metabolites do not have associated meta data.\n'.format(
                    len(missing_meta_data)))
                print('{0} metabolites have associated meta data.\n'.format(
                    len(set(self.metabolite_meta_df.index.values))))

            common_metabolites = list(
                set(self.metabolite_meta_df.index.values).intersection(set(self.metabolite_data.columns.values)))
            self.metabolite_meta_df = self.metabolite_meta_df.loc[common_metabolites]
        else:
            raise KeyError('No meta data file for metabolites in config file. Please include '
                           '"meta_data=<path/to/metabolite_meta_data/>" in config file under section "metabolite_data"')

    def get_metabolite_distance_matrix(self):
        print('\nObtaining metabolite distance matrix')
        if 'fingerprint_type' in self.config['metabolite_data'] and \
            self.config['metabolite_data']['fingerprint_type'].lower()=='infomax':
            self.metabolite_dist_df = pd.DataFrame(squareform(pdist(self.infomax_df.values)), index=self.infomax_df.index.values,
                                       columns=self.infomax_df.index.values)
        else:
            fingerprint_df = self.metabolite_meta_df['fingerprints']
            self.metabolite_dist_df = get_dist_mat(fingerprint_df.to_dict(), ftype=self.fingerprint)

        # self.metabolite_dist_df = 1-sim_df
        m_w_embs = list(set(self.metabolite_dist_df.columns.values).intersection(set(self.metabolite_data.columns.values)))
        m_wo_embs = list(set(self.metabolite_data.columns.values) - set(m_w_embs))
        self.metabolite_data = self.metabolite_data[m_w_embs + m_wo_embs]
        self.metabolite_dist_df=self.metabolite_dist_df[m_w_embs].loc[m_w_embs]
        print(f'{self.metabolite_dist_df.shape[0]}/{self.metabolite_data.shape[1]} metabolites have embedded locations')
        self.metabolite_dist_df.to_csv(self.save_path + '/met_dist.csv')
        fig,ax=plt.subplots(); ax.hist(self.metabolite_dist_df.values.flatten(), bins=20);ax.set_title(self.fingerprint)
        fig.savefig(self.save_path+'/met_dist.pdf')
        plt.close(fig)

    def get_metabolite_fingerprints(self):
        print('\nGetting metabolite fingerprints')
        if 'fingerprints' in self.config['metabolite_data'] and self.config['metabolite_data']['fingerprints']!='':
            with open(self.config['metabolite_data']['fingerprints'], 'rb') as f:
                fdict = pkl.load(f)
            fingerprint_series = pd.Series(fdict)
            fingerprint_series.name = 'fingerprints'
        else:
            if self.fingerprint is not None:
                if self.fingerprint.lower()=='rdkit' or self.fingerprint.lower()=='morgan' or self.fingerprint.lower()=='mqn':
                    fingerprint_series = get_rdkit_fingerprint(self.metabolite_meta_df[self.key_type.lower()],
                                                               self.fingerprint)
                elif self.fingerprint.lower()=='pubchem':
                    assert('fingerprints' in self.metabolite_meta_df.columns.values)
                    fingerprint_series = self.metabolite_meta_df['fingerprints'].squeeze()
                    fingerprint_series.name = 'fingerprints'
                elif self.fingerprint.lower()=='infomax':
                    self.infomax_df = get_infomax_fingerprint(self.metabolite_meta_df[self.key_type.lower()])
                    fingerprint_series = self.infomax_df
                elif self.fingerprint.lower()=='map4':
                    fingerprint_series = get_map4_fingerprint(self.metabolite_meta_df[self.key_type.lower()])
                    fingerprint_series.name = 'fingerprints'
            else:
                assert ('fingerprints' in self.metabolite_meta_df.columns.values)
                fingerprint_series = self.metabolite_meta_df['fingerprints'].squeeze()
                fingerprint_series.name = 'fingerprints'
            try:
                with open(self.save_path + '/fingerprints.pkl', 'wb') as f:
                    pkl.dump(fingerprint_series.to_dict(), f)
            except:
                pass
        if 'fingerprints' not in self.metabolite_meta_df.columns.values:
            self.metabolite_meta_df = self.metabolite_meta_df.join(fingerprint_series, how='left')

    def get_metabolite_intermediate_ids(self, key_type):
        print(f'\nGetting metabolite {key_type}')
        self.metabolite_meta_df = get_met_keys(self.metabolite_meta_df, key_type=key_type)
        self.metabolite_meta_df.to_csv(self.save_path + f'/metabolite_IDs.csv')

    def get_metabolite_classifications(self):
        print('\nGetting metabolite classfications')
        if 'inchikey' not in self.metabolite_meta_df.columns.values:
            meta_df_w_inchi=get_met_keys(self.metabolite_meta_df, key_type='inchikey')
            meta_df_w_inchi['inchikey'].to_csv(self.save_path+'/metabolite_InChIKey_only.csv')
        else:
            self.metabolite_meta_df['inchikey'].to_csv(self.save_path+'/metabolite_InChIKey_only.csv')
        os.system('Rscript --vanilla ./utilities/get_classy_fire.R "' + self.save_path + '/"' + ' metabolite_InChIKey_only.csv')
        self.metabolite_classifications = pd.read_csv(self.save_path+'/classy_fire_df.csv', header=0, index_col=0).T

    def get_metabolite_tree(self):
        self.metabolite_tree = get_metabolite_tree_from_classifications(self.metabolite_classifications)
        self.metabolite_tree.write(features=['name'], outfile=self.save_path+'/metabolite_tree.nhx', format=0)

        # colors = ete3.random_color(num=len(self.metabolite_classifications.columns.values))
        # i=0
        # for n in self.metabolite_tree.traverse():
        #     if n.is_leaf():
        #         n.add_face(ete3.TextFace(n.name, fgcolor=colors[i]), column=0, position='branch-top')
        #         n.name = ''
        #         i+=1
        #     # else:
        #     #     i += 1
        # ts = ete3.TreeStyle()
        # ts.show_leaf_name = True
        # try:
        #     self.metabolite_tree.render(self.save_path+'/classification_tree_for_IDd_metabolites.pdf', tree_style=ts)
        #     plt.close()
        # except:
        #     pass

    def get_filtering_parameters(self, preprocessing_section):
        preprocessing={}
        if 'limit_of_detection' in preprocessing_section and preprocessing_section[
            'limit_of_detection'] != '':
            preprocessing['lod'] = float(preprocessing_section['limit_of_detection'])
        if 'percent_present_in' in preprocessing_section and preprocessing_section[
            'percent_present_in'] != '':
            preprocessing['percent_present_in'] = float(
                preprocessing_section['percent_present_in'])
        if 'cov_percentile' in preprocessing_section and preprocessing_section[
            'cov_percentile']!='':
            preprocessing['cov_percentile'] = float(
                preprocessing_section['cov_percentile'])
        return preprocessing


    def split_data_from_covariate(self, data_dict):
        covar_nm = self.config['data']['covariate_variable']
        covar = self.subject_data[covar_nm]
        # specified_pos=False
        if 'covariate_positive_value' in self.config['data'] and self.config['data']['covariate_negative_value']!='':
            pos_val = self.config['data']['covariate_positive_value']
            if 'covariate_negative_value' not in self.config['data'] or self.config['data']['covariate_negative_value']=='':
                self.covar_bin = covar==pos_val
                self.codes = {1:pos_val}
            else:
                neg_val = self.config['data']['covariate_negative_value']
                covar = covar.loc[covar.isin([pos_val, neg_val])]
                self.covar_bin = covar==pos_val
                self.codes = {1:pos_val, 0:neg_val}
            # specified_pos = True
        else:
            vals_unique = pd.unique(covar.values)
            self.codes = {i:v for i,v in enumerate(vals_unique)}
            self.covar_bin = covar

        new_data_dicts={}
        print(f"Splitting data on the basis of {self.config['data']['covariate_variable']}")
        for cat in np.unique(self.covar_bin.values):
            # if specified_pos:
            #     if cat in self.codes.keys():
            #         new_key=self.codes[cat]
            #     else:
            #         new_key=f'not_{list(self.codes.keys())[0]}'
            # else:
            #     new_key = cat
            new_data_dicts[cat] = copy.deepcopy(data_dict)
            keep_samps = self.covar_bin.index.values[self.covar_bin==cat]
            new_data_dicts[cat]['X'] = data_dict['X'].loc[keep_samps]
            new_data_dicts[cat]['y'] = data_dict['y'].loc[keep_samps]
            # if cat in self.codes.keys():
            print(
                f"{len(keep_samps)} samples have {self.config['data']['covariate_variable']}={cat}")
            # else:
            #     print(f"{len(keep_samps)} samples have {self.config['data']['covariate_variable']}!={list(self.codes.keys())[0]}")
        return new_data_dicts

    def save_metabolite_data_to_pickle(self):
        # print(f'\nSaving metabolite data to {self.save_path + "/mets.pkl"}')
                     # 'preprocessing': compose(self.metabolite_processing_funcs)}
        ixs = [ix for ix in self.metabolite_data.index.values if ix in self.Y.index.values]
        self.Y = self.Y.loc[ixs]
        data_dict = {'X': self.metabolite_data, 'y': self.Y,
                     'distances': self.metabolite_dist_df,
                     'variable_tree': self.metabolite_tree,
                     'variable_names': self.metabolite_data.columns.values,
                     'preprocessing': self.metabolite_preprocessing,
                     'taxonomy':self.metabolite_classifications,
                     'labels':self.class_labels,
                     'metabolite_IDs':self.metabolite_meta_df}
        if 'collapse' in self.config['metabolite_data'] and self.config['metabolite_data'][
                        'collapse'].replace(' ', '').lower() == 'true':
            if self.collapse==True:
                data_dict = collapse_dataset(data_dict)
        if 'replicates' in self.config['metabolite_data'] and self.config['metabolite_data']['replicates']!='':
            replicates = read_ctsv(self.config['metabolite_data']['replicates'])
            data_dict['replicates'] = replicates
        if 'covariate_variable' in self.config['data'] and self.config['data']['covariate_variable']!='':
            data_dicts = self.split_data_from_covariate(data_dict)
            for k, d in data_dicts.items():
                if isinstance(k, str):
                    k = k.replace('.','-')
                else:
                    k = str(k)
                print(f'\nSaving metabolite data to {self.save_path + "/" + k + "_mets.pkl"}')
                with open(self.save_path +'/'+ k + '_mets.pkl', 'wb') as f:
                    pkl.dump(d, f)
        else:
            if 'name' in self.config['description'] and self.config['description']['name']!='':
                k = self.config['description']['name']
                print(f'\nSaving metabolite data to {self.save_path + "/" + k + "_mets.pkl"}')
                with open(self.save_path + '/' + k + '_mets.pkl', 'wb') as f:
                    pkl.dump(data_dict, f)
            else:
                print(f'\nSaving metabolite data to {self.save_path + "/" +"mets.pkl"}')
                with open(self.save_path+'/' + 'mets.pkl', 'wb') as f:
                    pkl.dump(data_dict, f)

    def save_sequence_data_to_pickle(self):
        print(f'\nSaving sequence data to {self.save_path + "/seqs.pkl"}')
        self.Y = self.Y.loc[self.sequence_data.index.values]
        data_dict = {'X': self.sequence_data, 'y': self.Y,
                     'distances': self.sequence_dist_df,
                     'variable_tree': self.sequence_tree,
                     'variable_names': self.sequence_data.columns.values,
                     'preprocessing': self.sequence_preprocessing,
                     'labels':self.class_labels}
        # 'preprocessing': compose(self.sequence_processing_funcs)}
        if self.taxonomy is not None:
            data_dict['taxonomy'] = self.taxonomy
        if self.sequences is not None:
            data_dict['sequences'] = self.sequences

        if 'replicates' in self.config['sequence_data'] and self.config['sequence_data']['replicates']!='':
            replicates = read_ctsv(self.config['sequence_data']['replicates'])
            data_dict['replicates'] = replicates

        if 'covariate_variable' in self.config['data'] and self.config['data']['covariate_variable']!='':
            data_dicts = self.split_data_from_covariate(data_dict)
            for k, data_dict in data_dicts.items():
                with open(self.save_path + '/' + str(k) + '_seqs.pkl', 'wb') as f:
                    pkl.dump(data_dict, f)
        else:
            if 'name' in self.config['description'] and self.config['description']['name']!='':
                with open(self.save_path + '/' + self.config['description']['name'] + '_seqs.pkl', 'wb') as f:
                    pkl.dump(data_dict, f)
            else:
                with open(self.save_path+'/' +'seqs.pkl', 'wb') as f:
                    pkl.dump(data_dict, f)




#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config_file', type=str, default='../config_files/cdi_TEST.cfg')
#     args = parser.parse_args()
#
#     # save_path = './datasets/'+args.config.split('./')[-1].split('/')[1].split('.')[0]+'/'
#     # print(save_path)
#     # if not os.path.isdir(save_path):
#     #     os.mkdir(save_path)
#     pData = ProcessData(args.config_file)






