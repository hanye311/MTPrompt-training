import torch
import yaml
import numpy as np
import pandas as pd
import csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import load_configs, calculate_class_weights
from transformers import AutoTokenizer, T5Tokenizer
from utils import truncate_seq
import os
import tqdm
import re
import esm
from esm import pretrained
from model import prepare_adapter_h_model
import pickle
from task_weighting import get_task_weights

# from esm.models.esmc import ESMC
# from esm.sdk.api import ESMProtein, LogitsConfig
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def find_indexes(string, chars, exclude_indexes):
    """
    Returns the 1-indexed positions of specified characters in a string, excluding certain positions.

    Parameters:
    - string (str): The input string in which to search for characters.
    - chars (list of str): A list of characters to search for in the string.
    - exclude_indexes (list of int): A list of 1-indexed positions to exclude from the search.

    Returns:
    - list of int: A list of 1-indexed positions where the characters from 'chars' appear in 'string',
                   excluding positions from 'exclude_indexes'.

    """
    # Convert the 1-indexed positions to 0-indexed positions for Python
    exclude_indexes = [i - 1 for i in exclude_indexes]

    indexes = []
    for i, char in enumerate(string):
        # Check if the character is in the chars list and not in the exclude list
        if char in chars and i not in exclude_indexes:
            # Convert the 0-indexed position back to 1-indexed for the result
            indexes.append(i + 1)
    return indexes


def check_ptm_site(sequence, positions, allowed_ptm_sites):
    for i in positions.copy():
        if i > len(sequence):
            positions.remove(i)
            continue
        elif not sequence[i - 1] in allowed_ptm_sites:
            positions.remove(i)
    return positions

def check_center_amino_acid(sequence, position, positive_amino_acids):
   if sequence[position - 1] in positive_amino_acids:
       return True

def extract_positions(sequence,configs):
    ptm_position= {}
    token_info = configs.encoder.condition_token.token_info
    for i in range(len(sequence)):
        if sequence[i] in token_info.keys():
            ptm_position[i]=token_info[sequence[i]]
        else:
            ptm_position[i]="None"

    return ptm_position


class PTMDataset(Dataset):
    def __init__(self, samples_list, configs):
        self.samples_list=samples_list
        self.configs = configs

        if  self.configs.encoder.model_name.startswith("esmc"):
            self.max_length = configs.encoder.max_len

            # self.client = ESMC.from_pretrained("esmc_600m").to(device)  # or "cpu"
        else:
            # self.esm2_model, self.esm2_alphabet = prepare_adapter_h_model(configs_all, logging)
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(configs.encoder.model_name)
            self.max_length = configs.encoder.max_len


    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        if self.configs.bam.model=='student':
            prot_id, sequence, label, teacher_distill_output,mask, task_token,positive_masks,negative_masks,index= (self.samples_list[index][0], self.samples_list[index][1], self.samples_list[index][2],
                                                                                      self.samples_list[index][3],self.samples_list[index][4],self.samples_list[index][5],
                                                                                      self.samples_list[index][6],self.samples_list[index][7],index)
            if self.configs.encoder.model_name.startswith("esmc"):
                encoded_sequence=sequence
                padded_label = np.array(label)
                # padded_mask= np.pad(mask, (0, self.max_length - len(label)), 'constant')
                padded_mask = np.array(mask)

            else:
                encoded_sequence = self.encoder_tokenizer(sequence, max_length=self.max_length + 2, padding='max_length',
                                                          truncation=True,
                                                          return_tensors="pt"
                                                          )

                encoded_sequence['input_ids'] = torch.squeeze(encoded_sequence['input_ids'])
                encoded_sequence['attention_mask'] = torch.squeeze(encoded_sequence['attention_mask'])

                padded_label = np.pad(label, (0, self.max_length - len(label)), 'constant')
                # padded_mask= np.pad(mask, (0, self.max_length - len(label)), 'constant')
                padded_mask = np.pad(mask, (0, self.max_length - len(label)), 'constant')
                padded_positive_masks = np.pad(positive_masks, (0, self.max_length - len(label)), 'constant')
                padded_negative_masks = np.pad(negative_masks, (0, self.max_length - len(label)), 'constant')
            return prot_id, encoded_sequence,padded_label,teacher_distill_output,padded_mask.astype(bool),task_token,padded_positive_masks.astype(bool),padded_negative_masks.astype(bool),index
        else:
            prot_id, sequence, label, mask, task_token,positive_masks,negative_masks, index = (
            self.samples_list[index][0], self.samples_list[index][1], self.samples_list[index][2],
            self.samples_list[index][3], self.samples_list[index][4], self.samples_list[index][5], self.samples_list[index][6],index)
            if  self.configs.encoder.model_name.startswith("esmc"):
                encoded_sequence = sequence

                padded_label = np.array(label)
                # padded_mask= np.pad(mask, (0, self.max_length - len(label)), 'constant')
                padded_mask = np.array(mask)
            else:
                encoded_sequence = self.encoder_tokenizer(sequence, max_length=self.max_length + 2, padding='max_length',
                                                          truncation=True,
                                                          return_tensors="pt"
                                                          )

                encoded_sequence['input_ids'] = torch.squeeze(encoded_sequence['input_ids'])
                encoded_sequence['attention_mask'] = torch.squeeze(encoded_sequence['attention_mask'])

                padded_label = np.pad(label, (0, self.max_length - len(label)), 'constant')
                # padded_mask = np.pad(mask, (0, self.max_length - len(label)), 'constant')
                padded_mask = np.pad(mask, (0, self.max_length - len(label)), 'constant')
                padded_positive_masks = np.pad(positive_masks, (0, self.max_length - len(label)), 'constant')
                padded_negative_masks = np.pad(negative_masks, (0, self.max_length - len(label)), 'constant')
            return prot_id, encoded_sequence, padded_label, padded_mask.astype(bool), task_token, padded_positive_masks.astype(bool),padded_negative_masks.astype(bool),index


def prepare_task(dataset_path, task_token, positive_amino_acids, max_length, logging,configs,type,task_name):
    df = pd.read_csv(dataset_path)

    removed_indices = []
    for row in tqdm.tqdm(df.itertuples(), total=len(df), disable=True):
        sequence = row.Sequence
        if len(sequence) >max_length - 2:
            removed_indices.append(row[0])

    df.drop(removed_indices, inplace=True)   #todo

    logging.info(f'{task_token}: number of removed sequences bigger than {max_length - 2}: {len(removed_indices)}')
    logging.info(f'{task_token}: number left: {df.shape[0]}')
    sample_weights = 1
    samples = []
    sum_number_positives=0
    sum_number_negatives=0
    if configs.bam.model=='student':
        distill_output_path=configs.bam.teacher_distill_output_path

        # Load tensor from the pickle file
        with open(os.path.join(distill_output_path,task_name,type,"predictions.pkl"), 'rb') as f:
            teacher_distill_output = pickle.load(f)
            # teacher_distill_output=torch.load(f, map_location=torch.device('cpu'))


    for row in df.itertuples():
        positions = list(set([int(p) for p in re.findall(r'\b\d+\b', row.Position)]))
        positions = sorted(list(set(positions)))
        sequence = row.Sequence
        # ptm_position=extract_positions(sequence,configs)

        prot_id = row.Uniprotid
        positive_positions = check_ptm_site(sequence, positions, positive_amino_acids)
        label = [0] * len(sequence)

        # Replace the zeros with ones at the positions of modified residues
        for position in positive_positions:
            # Subtracting 1 because positions are 1-indexed, but lists are 0-indexed
            label[position - 1] = 1

        valid_mask=label.copy()
        valid_mask_positive = [0] * len(sequence)
        valid_mask_negative = [0] * len(sequence)

        negative_positions = find_indexes(sequence, positive_amino_acids, positive_positions)
        all_positions = sorted(negative_positions + positive_positions)
        sum_number_positives=len(positive_positions)+sum_number_positives
        sum_number_negatives=len(negative_positions)+sum_number_negatives
        for position in positive_positions:
            # Subtracting 1 because positions are 1-indexed, but lists are 0-indexed
            valid_mask_positive[position - 1] = 1

        for position in negative_positions:
            # Subtracting 1 because positions are 1-indexed, but lists are 0-indexed
            valid_mask_negative[position - 1] = 1

        for position in all_positions:
            # Subtracting 1 because positions are 1-indexed, but lists are 0-indexed
            valid_mask[position - 1] = 1

        if configs.bam.model=='student':
            if type=="train":
                samples.append((prot_id,sequence,label,teacher_distill_output[prot_id],valid_mask,task_token,valid_mask_positive,valid_mask_negative))
            else:
                samples.append((prot_id, sequence, label, "none", valid_mask, task_token,
                                valid_mask_positive, valid_mask_negative))
        else:
            samples.append((prot_id, sequence, label, valid_mask, task_token,valid_mask_positive,valid_mask_negative))
    #
    # samples = random_pick(samples, max_samples, 42)

    # logging.info(f'{task_token}: remaining train samples: {len(samples)}')
    print(sum_number_positives,sum_number_negatives)
    return samples

def get_train_samples(configs, logging,task_info):
    train_samples = prepare_task(
        dataset_path=os.path.join(configs.train_settings.data_path, task_info['file_name']),
        task_token=task_info['id'],
        positive_amino_acids=task_info['ptm_amino_acid'],
        max_length=configs.encoder.max_len,
        # max_samples=configs_all.train_settings.max_task_samples,
        logging=logging,
        configs=configs,
        type="train",
        task_name=task_info["task_name"]
    )

    return train_samples

def get_val_samples(configs, logging,task_info):
    val_samples = prepare_task(
        dataset_path = os.path.join(configs.valid_settings.data_path, task_info['file_name']),
        task_token = task_info['id'],
        positive_amino_acids=task_info['ptm_amino_acid'],
        max_length=configs.encoder.max_len,
        # max_samples=configs_all.train_settings.max_task_samples,
        logging=logging,
        configs=configs,
        type="valid",
        task_name=task_info["task_name"]
    )
    dataset_val = PTMDataset(val_samples, configs)
    val_loader = DataLoader(dataset_val, batch_size=configs.valid_settings.batch_size,
                            shuffle=True, pin_memory=False, drop_last=False,
                            num_workers=configs.valid_settings.num_workers)

    return val_loader

def get_test_samples(configs, logging,task_info):
    test_samples = prepare_task(
        dataset_path=os.path.join(configs.test_settings.data_path, task_info['file_name']),
        task_token=task_info['id'],
        positive_amino_acids=task_info['ptm_amino_acid'],
        max_length=configs.encoder.max_len,
        # max_samples=configs_all.train_settings.max_task_samples,
        logging=logging,
        configs=configs,
        type="test",
        task_name=task_info["task_name"]
    )
    dataset_test = PTMDataset(test_samples, configs)
    test_loader = DataLoader(dataset_test, batch_size=configs.test_settings.batch_size,
                             shuffle=True, pin_memory=False, drop_last=False,
                             num_workers=configs.test_settings.num_workers)

    return test_loader

def prepare_dataloaders_ptm(configs,logging):

    # task_dic= {}
    task_list=[]

    dataloaders_dict_val = {}
    dataloaders_dict_test= {}
    dataloaders_train_list=[]



    if configs.tasks.Phosphorylation_S==True:
        task_list.append(
            {'task_name':"Phosphorylation_S",'id':configs.task_ids.Phosphorylation_S,'file_name':"final_Phosphorylation_S.csv",
                          'ptm_amino_acid':["S"]})

    if configs.tasks.Phosphorylation_T==True:
        task_list.append(
            {'task_name': "Phosphorylation_T", 'id': configs.task_ids.Phosphorylation_T, 'file_name': "final_Phosphorylation_T.csv",
             'ptm_amino_acid': ["T"]})

    if configs.tasks.Phosphorylation_Y==True:
        task_list.append(
            {'task_name': "Phosphorylation_Y", 'id': configs.task_ids.Phosphorylation_Y, 'file_name': "final_Phosphorylation_Y.csv",
             'ptm_amino_acid': ["Y"]})

    if configs.tasks.Ubiquitination_K==True:
        task_list.append(
            {'task_name': "Ubiquitination_K", 'id': configs.task_ids.Ubiquitination_K, 'file_name': "final_Ubiquitination_K.csv",
             'ptm_amino_acid': ["K"]})

    if configs.tasks.Acetylation_K==True:
        task_list.append(
            {'task_name': "Acetylation_K", 'id': configs.task_ids.Acetylation_K, 'file_name': "final_Acetylation_K.csv",
             'ptm_amino_acid': ["K"]})

    if configs.tasks.OlinkedGlycosylation_S==True:
        task_list.append(
            {'task_name': "OlinkedGlycosylation_S", 'id': configs.task_ids.OlinkedGlycosylation_S, 'file_name': "final_OlinkedGlycosylation_S.csv",
             'ptm_amino_acid': ["S"]})

    if configs.tasks.Methylation_R==True:
        task_list.append(
            {'task_name': "Methylation_R", 'id': configs.task_ids.Methylation_R, 'file_name': "final_Methylation_R.csv",
             'ptm_amino_acid': ["R"]})

    if configs.tasks.NlinkedGlycosylation_N==True:
        task_list.append(
            {'task_name': "NlinkedGlycosylation_N", 'id': configs.task_ids.NlinkedGlycosylation_N, 'file_name': "final_NlinkedGlycosylation_N.csv",
             'ptm_amino_acid': ["N"]})

    if configs.tasks.OlinkedGlycosylation_T==True:
        task_list.append(
            {'task_name': "OlinkedGlycosylation_T", 'id': configs.task_ids.OlinkedGlycosylation_T, 'file_name': "final_OlinkedGlycosylation_T.csv",
             'ptm_amino_acid': ["T"]})

    if configs.tasks.Methylation_K==True:
        task_list.append(
            {'task_name': "Methylation_K", 'id': configs.task_ids.Methylation_K, 'file_name': "final_Methylation_K.csv",
             'ptm_amino_acid': ["K"]})

    if configs.tasks.Palmitoylation_C==True:
        task_list.append(
            {'task_name': "Palmitoylation_C", 'id': configs.task_ids.Palmitoylation_C, 'file_name': "final_Palmitoylation_C.csv",
             'ptm_amino_acid': ["C"]})

    if configs.tasks.Sumoylation_K==True:
        task_list.append(
            {'task_name': "Sumoylation_K", 'id': configs.task_ids.Sumoylation_K, 'file_name': "final_Sumoylation_K.csv",
             'ptm_amino_acid': ["K"]})

    if configs.tasks.Succinylation_K==True:
        task_list.append(
            {'task_name': "Succinylation_K", 'id': configs.task_ids.Succinylation_K, 'file_name': "final_Succinylation_K.csv",
             'ptm_amino_acid': ["K"]})


    data_sizes = {
        0: 10002,
        1: 7330,
        2: 3257,
        3: 997,
        4: 5172,
        5: 808,
        6: 1414,
        7: 9622,
        8: 591,
        9: 481,
        10: 1323,
        11: 3185,
        12: 1776
    }
    multiples, task_weights = get_task_weights(configs, data_sizes)

    for task_info in task_list:
        sample_list = get_train_samples(configs, logging, task_info)
        # data_sizes[task_info['id']]=len(sample_list)
        if configs.bam.dataset_multiples:
            for multiple_time in range(multiples[task_info['id']]):
                dataloaders_train_list.extend(sample_list)
        else:
            dataloaders_train_list.extend(sample_list)
        dataloaders_dict_val[task_info['task_name']] = get_val_samples(configs, logging, task_info)
        dataloaders_dict_test[task_info['task_name']] = get_test_samples(configs, logging, task_info)


    dataset_train = PTMDataset(dataloaders_train_list, configs)

    train_loader = DataLoader(dataset_train, batch_size=configs.train_settings.batch_size,
                              shuffle=True, pin_memory=False, drop_last=False,
                              num_workers=configs.train_settings.num_workers)
    # task_weights=get_task_weights(configs,data_sizes)
    return {'train': train_loader, 'valid': dataloaders_dict_val, 'test': dataloaders_dict_test},task_weights

if __name__ == '__main__':
    config_path = './config_enzyme_reaction.yaml'
    with open(config_path) as file:
        configs_dict = yaml.full_load(file)

    configs_file = load_configs(configs_dict)

    dataloaders_dict = prepare_dataloaders_enzyme_reaction(configs_file)
    max_position_value = []
    amino_acid = []
    for batch in dataloaders_dict['train']:
        sequence_batch, label_batch, position_batch, weights_batch = batch
        # print(sequence_batch['input_ids'].shape)
        # print(label_batch.shape)
        # print(position_batch.shape)
        max_position_value.append(position_batch.squeeze().numpy().item())
        amino_acid.append(sequence_batch["input_ids"][0][position_batch.squeeze().numpy().item()].item())
    print(set(max_position_value))
    print([dataloaders_dict['train'].dataset.encoder_tokenizer.id_to_token(i) for i in set(amino_acid)])
    print('done')
