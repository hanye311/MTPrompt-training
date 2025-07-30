import torch
import yaml
from torch import nn
from torch.nn import functional as F
from collections.abc import Sequence
from transformers import EsmModel, T5Tokenizer, T5Model
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import load_configs, get_dummy_logging
import esm_adapterH
import esm
import numpy as np
import copy
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from typing import Mapping, Optional, Tuple, Any, Union
from torch import nn, Tensor
import os




from esm_adapterH.prompt_tuning import PrefixTuning

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def verify_data_types(model, logging=None):
    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        if logging:
            logging.info(f"{k}, {v}, {v / total}")


def get_task_embedding(configs):

    task_embeddings_dic = {}


    output_data_folder = configs.encoder.prompt.task_token_path


    if configs.tasks.Phosphorylation_S == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Phosphorylation_S_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Phosphorylation_S, task_embedding)
        # task_embeddings_list.append(task_embedding)
    if configs.tasks.Phosphorylation_T == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Phosphorylation_T_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Phosphorylation_T, task_embedding)
        # task_embeddings_list.append(task_embedding)
    if configs.tasks.Phosphorylation_Y == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Phosphorylation_Y_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Phosphorylation_Y, task_embedding)
        # task_embeddings_list.append(task_embedding)
    if configs.tasks.Ubiquitination_K == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Ubiquitination_K_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Ubiquitination_K, task_embedding)
        # task_embeddings_list.append(task_embedding)
    if configs.tasks.Acetylation_K == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Acetylation_K_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Acetylation_K, task_embedding)
        # task_embeddings_list.append(task_embedding)
    if configs.tasks.OlinkedGlycosylation_S == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "OlinkedGlycosylation_S_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.OlinkedGlycosylation_S, task_embedding)
        # task_embeddings_list.append(task_embedding)
    if configs.tasks.Methylation_R == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Methylation_R_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Methylation_R, task_embedding)
        # task_embeddings_list.append(task_embedding)
    if configs.tasks.NlinkedGlycosylation_N == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "NlinkedGlycosylation_N_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.NlinkedGlycosylation_N, task_embedding)
        # task_embeddings_list.append(task_embedding)
    if configs.tasks.OlinkedGlycosylation_T == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "OlinkedGlycosylation_T_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.OlinkedGlycosylation_T, task_embedding)

    if configs.tasks.Methylation_K == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Methylation_K_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Methylation_K, task_embedding)
        # task_embeddings_list.append(task_embedding)
    if configs.tasks.Palmitoylation_C == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Palmitoylation_C_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Palmitoylation_C, task_embedding)

    if configs.tasks.Sumoylation_K == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Sumoylation_K_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Sumoylation_K, task_embedding)

    if configs.tasks.Succinylation_K == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Succinylation_K_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Succinylation_K, task_embedding)

    if configs.tasks.Amidation_C == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Amidation_C_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Amidation_C, task_embedding)

    if configs.tasks.Amidation_F == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Amidation_F_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Amidation_F, task_embedding)

    if configs.tasks.Amidation_L == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Amidation_L_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Amidation_L, task_embedding)

    if configs.tasks.Amidation_V == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Amidation_V_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Amidation_V, task_embedding)

    if configs.tasks.Hydroxylation_K == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Hydroxylation_K_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Hydroxylation_K, task_embedding)

    if configs.tasks.Hydroxylation_P == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Hydroxylation_P_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Hydroxylation_P, task_embedding)

    if configs.tasks.Methylation_C == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Methylation_C_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Methylation_C, task_embedding)

    if configs.tasks.Phosphorylation_K == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Phosphorylation_K_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Phosphorylation_K, task_embedding)

    if configs.tasks.Pyrrolidone_carboxylic_acid_E == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Pyrrolidone_carboxylic_acid_E_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Pyrrolidone_carboxylic_acid_E, task_embedding)

    if configs.tasks.Pyrrolidone_carboxylic_acid_Q == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Pyrrolidone_carboxylic_acid_Q_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Pyrrolidone_carboxylic_acid_Q, task_embedding)

    if configs.tasks.S_nitrosocysteine_C == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "S-nitrosocysteine_C_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.S_nitrosocysteine_C, task_embedding)

    if configs.tasks.Sulfation_Y == True:
        task_embedding = torch.load(
            os.path.join(output_data_folder, "Sulfation_Y_cluster_embeddings_positive.pt"))
        task_embeddings_dic.setdefault(configs.task_ids.Sulfation_Y, task_embedding)

    return task_embeddings_dic


def prepare_adapter_h_model(configs, logging=None):
    if logging:
        logging.info("use adapterH ESM model")

    adapter_args = configs.encoder.adapter_h
    model_name = configs.encoder.model_name.split('/')[-1]

    # Create the model dynamically using module attributes
    model_constructor = getattr(esm_adapterH.pretrained, model_name, None)
    model, alphabet = model_constructor(adapter_args)
    num_layers = model.num_layers
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    if configs.encoder.prompt.enable:
        if not hasattr(configs.encoder.prompt, "num_tasks"):
            configs.encoder.prompt.num_tasks = 1

        task_embedding_dic = get_task_embedding(configs)  ##get pretrain task token

        model.prefix_module = PrefixTuning(configs, task_embedding_dic, model,
                                           prompt_len=configs.encoder.prompt.prompt_len,
                                           prompt_layer_indices=configs.encoder.prompt.prompt_layer_indices,
                                           num_tasks=configs.encoder.prompt.num_tasks
                                           )
        if configs.encoder.prompt.if_grads:
            for param in model.prefix_module.parameters():
                param.requires_grad = True
        else:
            for param in model.prefix_module.parameters():
                param.requires_grad = False

    #     model.prefix_module.prompt_layer_dict["layer_0"]["0"].requires_grad=False
    if configs.encoder.adapter_h.enable:
        if not isinstance(configs.encoder.adapter_h.freeze_adapter_layers, list):
            configs.encoder.adapter_h.freeze_adapter_layers = [configs.encoder.adapter_h.freeze_adapter_layers]

    if configs.encoder.fine_tune.enable:
        if not isinstance(configs.encoder.fine_tune.freeze_adapter_layers, list):
            configs.encoder.fine_tune.freeze_adapter_layers = [configs.encoder.fine_tune.freeze_adapter_layers]

    if configs.encoder.lora.enable:
        if logging:
            logging.info('enable LoRa on top of adapterH model')
        if hasattr(configs.encoder.lora, "lora_targets"):
            lora_targets = configs.encoder.lora.lora_targets
        else:
            lora_targets = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                            "self_attn.out_proj"]
        target_modules = []
        if configs.encoder.lora.esm_num_end_lora > 0:
            start_layer_idx = np.max([num_layers - configs.encoder.lora.esm_num_end_lora, 0])
            for idx in range(start_layer_idx, num_layers):
                for layer_name in lora_targets:
                    target_modules.append(f"layers.{idx}.{layer_name}")

        config = LoraConfig(
            r=configs.encoder.lora.r,
            lora_alpha=configs.encoder.lora.lora_alpha,
            target_modules=target_modules,
            inference_mode=False,
            lora_dropout=configs.encoder.lora.lora_dropout,
            bias="none",
            # modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, config)

        verify_data_types(model, logging)

    elif not configs.encoder.lora.enable and configs.encoder.fine_tune.enable:
        # fine-tune the latest layer

        # Allow the parameters of the last transformer block to be updated during fine-tuning
        for param in model.layers[-configs.encoder.fine_tune.last_layers_trainable:].parameters():
            param.requires_grad = True

        # if you need fine-tune last layer, the emb_layer_norm_after for last representation should be updated
        if configs.encoder.fine_tune.last_layers_trainable != 0:
            for param in model.emb_layer_norm_after.parameters():
                param.requires_grad = True

    # only freeze all the parameters once at the beginning. then open some layers later
    # only make adapterH trainable according to freeze_adapter_layers
    if configs.encoder.adapter_h.enable:
        for adapter_idx, value in enumerate(configs.encoder.adapter_h.freeze_adapter_layers):
            if not value:
                for name, param in model.named_parameters():
                    adapter_name = f"adapter_{adapter_idx}"
                    if adapter_name in name:
                        # Freeze all parameters by default
                        param.requires_grad = True

    # only freeze all the parameters once at the beginning. then open some layers later,but because
    # of fine_tune, adapter layers might be tunable.
    # change on 1/15/2024 not need to use freeze_adapter_layers to control fine-tune part! use another parameter instead and must after setting of freeze_adapter_layers
    if configs.encoder.fine_tune.enable:  # only see fine_tune.freeze_adapter_layers when fine-tune is available
        for adapter_idx, value in enumerate(configs.encoder.fine_tune.freeze_adapter_layers):
            if value:
                for name, param in model.named_parameters():
                    adapter_name = f"adapter_{adapter_idx}"
                    if adapter_name in name:
                        # Freeze all parameters by default
                        print("freeze adapter in fine-tune")
                        param.requires_grad = False
    # """

    if configs.encoder.tune_embedding:
        for param in model.embed_tokens.parameters():
            param.requires_grad = True

    # if configs.encoder.prompt.enable:
    #     for param in model.prefix_module.parameters():
    #         param.requires_grad = True
    if configs.encoder.prompt.enable:
        if configs.encoder.prompt.if_grads:
            for param in model.prefix_module.parameters():
                param.requires_grad = True
        else:
            for param in model.prefix_module.parameters():
                param.requires_grad = False

    return model, alphabet


def prepare_esm_model(configs, logging=None):
    if logging:
        logging.info("use ESM model")
    #
    # model_name=str(configs.encoder.model_name)
    # client = ESMC.from_pretrained(model_name).to(device)  # or "cpu"

    model_name = configs.encoder.model_name.split('/')[-1]

    # Create the model dynamically using module attributes
    model_constructor = getattr(esm.pretrained, model_name, None)
    model, alphabet = model_constructor()
    num_layers = model.num_layers
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

        # only freeze all the parameters once at the beginning. then open some layers later

    if configs.encoder.lora.enable:
        if logging:
            logging.info('enable LoRa on top of esm model')
        # target_modules = [
        #    "k_proj", "v_proj", "q_proj","fc1", "fc2"]
        if hasattr(configs.encoder.lora, "lora_targets"):
            lora_targets = configs.encoder.lora.lora_targets
        else:
            lora_targets = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                            "self_attn.out_proj"]
        target_modules = []
        if configs.encoder.lora.esm_num_end_lora > 0:
            start_layer_idx = np.max([num_layers - configs.encoder.lora.esm_num_end_lora, 0])
            for idx in range(start_layer_idx, num_layers):
                for layer_name in lora_targets:
                    target_modules.append(f"layers.{idx}.{layer_name}")

        config = LoraConfig(
            r=configs.encoder.lora.r,
            lora_alpha=configs.encoder.lora.lora_alpha,
            target_modules=target_modules,
            inference_mode=False,
            lora_dropout=configs.encoder.lora.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, config)

        verify_data_types(model, logging)

    elif not configs.encoder.lora.enable and configs.encoder.fine_tune.enable:
        # fine-tune the latest layer
        # Allow the parameters of the last transformer block to be updated during fine-tuning
        for param in model.layers[-configs.encoder.fine_tune.last_layers_trainable:].parameters():
            param.requires_grad = True

        # if you need fine-tune last layer, the emb_layer_norm_after for last representation should be updated
        if configs.encoder.fine_tune.last_layers_trainable != 0:
            for param in model.emb_layer_norm_after.parameters():
                param.requires_grad = True
    elif hasattr(configs.encoder, "prompt"):

        if configs.encoder.prompt.enable:
            if not hasattr(configs.encoder.prompt, "num_tasks"):
                configs.encoder.prompt.num_tasks = 1

            model.prefix_module = PrefixTuning(model, prompt_len=configs.encoder.prompt.prompt_len,
                                               prompt_layer_indices=configs.encoder.prompt.prompt_layer_indices,
                                               # num_tasks = configs.encoder.prompt.num_tasks
                                               )
            for param in model.prefix_module.parameters():
                param.requires_grad = True

    if configs.encoder.tune_embedding:
        if logging:
            logging.info('make esm embedding parameters trainable')

        for param in model.embed_tokens.parameters():
            param.requires_grad = True

    return model, alphabet


class MoBYMLP(nn.Module):
    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(MoBYMLP, self).__init__()

        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))  # relu cannot be used with sigmoid!!! smallest will be 0.5?
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Linear(in_dim if num_layers == 1 else inner_dim,
                                    out_dim) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        # print("mlp forward")
        # print(x.shape)  #[128,512,100]
        batch_size, seqlen, features_dim = x.shape
        x = x.reshape(-1, features_dim)
        x_hidden = self.linear_hidden(x)
        x = self.linear_out(x_hidden)

        x = x.reshape(batch_size, seqlen, -1)
        return x, x_hidden


class MoBYMLP_multihead(nn.Module):
    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2, num_tasks=1):
        super(MoBYMLP_multihead, self).__init__()

        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))  # relu cannot be used with sigmoid!!! smallest will be 0.5?
        self.linear_hidden = nn.Sequential(*linear_hidden)


        self.task_heads = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_tasks)])


    def forward(self, x):

        batch_size, seqlen, features_dim = x.shape
        x = x.reshape(-1, features_dim)
        x = self.linear_hidden(x)
        task_outputs = [head(x).reshape(batch_size, seqlen, -1) for head in
                        self.task_heads]  # List of outputs for each task



        return task_outputs, x


class MoBYMLP_Flattern(nn.Module):
    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(MoBYMLP_Flattern, self).__init__()

        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Linear(in_dim * 21 if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))  # relu cannot be used with sigmoid!!! smallest will be 0.5?
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Linear(in_dim if num_layers == 1 else inner_dim,
                                    out_dim) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        # print("mlp forward")
        # print(x.shape)  #[128,512,100]
        batch_size, seqlen, features_dim = x.shape
        x = x.reshape(batch_size, -1)
        x = self.linear_hidden(x)
        x = self.linear_out(x)

        # x = x.reshape(batch_size,seqlen,-1)
        return x


class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels, configs, stride=1, padding="same"):
        super(MultiScaleCNN, self).__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels, configs.projector.out_channels, configs.projector.kernel_sizes[i], stride, padding)
            for i in range(len(configs.projector.kernel_sizes))
        ])
        self.batchnorm1 = nn.BatchNorm1d(configs.projector.out_channels * len(
            configs.projector.kernel_sizes))  # BatchNorm after the first Conv layer
        self.conv_layers2 = nn.ModuleList([
            nn.Conv1d(configs.projector.out_channels * len(configs.projector.kernel_sizes),
                      configs.projector.out_channels, configs.projector.kernel_sizes[i], stride, padding) for i in
            range(len(configs.projector.kernel_sizes))
        ])
        self.batchnorm2 = nn.BatchNorm1d(configs.projector.out_channels * len(
            configs.projector.kernel_sizes))  # BatchNorm after the second Conv layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=configs.projector.droprate)
        # self.fc_shared = nn.Linear(out_channels * len(kernel_sizes), inner_linear_dim)  # Shared fully connected layer
        # self.fc_multiclass = nn.Linear(inner_linear_dim, output_dim)  # Output layer for multi-class classification
        # use on 7.25.2024 4:19
        if configs.projector.if_multihead == True:
            self.fc_multiclass = MoBYMLP_multihead(
                in_dim=configs.projector.out_channels * len(configs.projector.kernel_sizes),
                inner_dim=configs.projector.out_channels * len(configs.projector.kernel_sizes),
                out_dim=configs.projector.output_dim, num_layers=configs.projector.num_layers,
                num_tasks=configs.encoder.prompt.num_tasks)
        else:
            if configs.projector.if_flattern == True:
                self.fc_multiclass = MoBYMLP_Flattern(
                    in_dim=configs.projector.out_channels * len(configs.projector.kernel_sizes),
                    inner_dim=configs.projector.out_channels * len(configs.projector.kernel_sizes),
                    out_dim=configs.projector.output_dim, num_layers=configs.projector.num_layers)
            else:
                self.fc_multiclass = MoBYMLP(
                    in_dim=configs.projector.out_channels * len(configs.projector.kernel_sizes),
                    inner_dim=configs.projector.inner_linear_dim,
                    out_dim=configs.projector.output_dim, num_layers=configs.projector.num_layers)
        self.configs = configs

    def forward(self, x):

        # x  batch length input channel
        x = x.permute(0, 2, 1)
        # x=x.unsqueeze(2)

        conv_results = [conv_layer(x) for conv_layer in self.conv_layers]
        x = torch.cat(conv_results, dim=1)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        conv_results2 = [conv_layer(x) for conv_layer in self.conv_layers2]
        x = torch.cat(conv_results2, dim=1)
        x = self.batchnorm2(x)
        x = self.relu(x)

        x_contact = x.permute(0, 2, 1)  # batch, length, out_channels
        # x = self.fc_shared(x)
        # print(x.shape)
        x, x_hidden = self.fc_multiclass(x_contact)  # batch, length, output_dim
        return x, x_hidden


# Define Positional Encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no batch normalization, activation or dropout in the last layer.

    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input):
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        # hidden = F.softmax(hidden, dim=-1)
        return hidden


def prepare_configs_mergedESM2(configs):
    merged_configs = copy.deepcopy(configs)
    # if has tune_embedding in merge2ESM2 use this specific config, if not, share with original configs_all
    if hasattr(configs.encoder.merge2ESM2, "tune_embedding"):
        merged_configs.encoder.tune_embedding = configs.encoder.merge2ESM2.tune_embedding
    if hasattr(configs.encoder.merge2ESM2, "fine_tune"):
        merged_configs.encoder.fine_tune = configs.encoder.merge2ESM2.fine_tune
    if hasattr(configs.encoder.merge2ESM2, "lora"):
        merged_configs.encoder.lora = configs.encoder.merge2ESM2.lora
    if hasattr(configs.encoder.merge2ESM2, "adapter_h"):
        merged_configs.encoder.adapter_h = configs.encoder.merge2ESM2.adapter_h

    return merged_configs


class EsmClassificationHeadMHACustomCNN(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, configs):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True)
        # self.dense = nn.Linear(input_dim, input_dim)
        # self.dropout = nn.Dropout(0.3)
        # self.out_proj = nn.Linear(input_dim, 2)

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_dim, configs.projector.out_channels, configs.projector.kernel_sizes[i], stride=1,
                      padding='same')
            for i in range(len(configs.projector.kernel_sizes))
        ])
        self.batchnorm1 = nn.BatchNorm1d(configs.projector.out_channels * len(
            configs.projector.kernel_sizes))  # BatchNorm after the first Conv layer
        self.conv_layers2 = nn.ModuleList([
            nn.Conv1d(configs.projector.out_channels * len(configs.projector.kernel_sizes),
                      configs.projector.out_channels, configs.projector.kernel_sizes[i], stride=1, padding='same') for i
            in
            range(len(configs.projector.kernel_sizes))
        ])
        self.batchnorm2 = nn.BatchNorm1d(configs.projector.out_channels * len(
            configs.projector.kernel_sizes))  # BatchNorm after the second Conv layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=configs.projector.droprate)
        # self.fc_shared = nn.Linear(out_channels * len(kernel_sizes), inner_linear_dim)  # Shared fully connected layer
        # self.fc_multiclass = nn.Linear(inner_linear_dim, output_dim)  # Output layer for multi-class classification
        # use on 7.25.2024 4:19
        if configs.projector.if_flattern == True:
            self.fc_multiclass = MoBYMLP_Flattern(
                in_dim=configs.projector.out_channels * len(configs.projector.kernel_sizes),
                inner_dim=configs.projector.out_channels * len(configs.projector.kernel_sizes),
                out_dim=configs.projector.output_dim, num_layers=configs.projector.num_layers)
        else:
            self.fc_multiclass = MoBYMLP(in_dim=configs.projector.out_channels * len(configs.projector.kernel_sizes),
                                         inner_dim=configs.projector.inner_linear_dim,
                                         out_dim=configs.projector.output_dim, num_layers=configs.projector.num_layers)
        self.configs = configs

    def forward(self, features, **kwargs):
        x_task_token = features[:, :500, :]

        x_sequence = features[:, 500:, :]
        attn_output, attn_output_weights = self.multihead_attn(x_sequence, x_task_token, x_task_token)
        # attn_output, attn_output_weights = self.multihead_attn(x_task_token, x_sequence, x_sequence)
        # x = attn_output[:, 10, :]  # take center substrate token

        # x  batch length input channel
        x = attn_output.permute(0, 2, 1)
        # x=x.unsqueeze(2)

        conv_results = [conv_layer(x) for conv_layer in self.conv_layers]
        x = torch.cat(conv_results, dim=1)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        conv_results2 = [conv_layer(x) for conv_layer in self.conv_layers2]
        x = torch.cat(conv_results2, dim=1)
        x = self.batchnorm2(x)
        x = self.relu(x)

        x = x.permute(0, 2, 1)  # batch, length, out_channels

        x = self.fc_multiclass(x)  # batch, length, output_dim


        return x


class EsmClassificationHeadMHACustom(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, config):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True)
        self.dense = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(0.3)
        self.out_proj = nn.Linear(input_dim, 2)

    def forward(self, features, **kwargs):
        x_task_token = features[:, :21, :]

        x_sequence = features[:, 21:, :]
        attn_output, attn_output_weights = self.multihead_attn(x_sequence, x_task_token, x_task_token)
        # attn_output, attn_output_weights = self.multihead_attn(x_task_token, x_sequence, x_sequence)
        # x = attn_output[:, 10, :]  # take center substrate token

        x = self.dropout(attn_output)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        # return x, attn_output, attn_output_weights
        return x


class EncoderSSPTM(nn.Module):
    def __init__(self, logging, configs):
        super().__init__()
        if configs.encoder.adapter_h.enable:
            self.esm2, self.alphabet = prepare_adapter_h_model(configs, logging)
        else:
            # self.esm2, self.alphabet = prepare_esm_model(configs, logging)
            self.esm2 = prepare_esm_model(configs, logging)
        self.configs = configs
        # extract the embedding size
        if self.configs.encoder.model_name.startswith("esmc"):
            mlp_input_dim = self.esm2.embed.embedding_dim
        else:
            mlp_input_dim = self.esm2.embed_dim

        if configs.projector.projector_type == 'MLP':

            mlp_hidden_dim = configs.encoder.mlp_hidden_dim
            hidden_dims = [mlp_hidden_dim] * (configs.encoder.mlp_layer_num - 1)
            self.mlp = MultiLayerPerceptron(mlp_input_dim, hidden_dims + [configs.encoder.num_classes],
                                            batch_norm=False,
                                            dropout=configs.encoder.head_dropout)

        elif configs.projector.projector_type == 'CNN':

            in_channels = mlp_input_dim
            self.mlp = MultiScaleCNN(in_channels, configs)

        elif configs.projector.projector_type == "MHACustom":
            self.mlp = EsmClassificationHeadMHACustom(mlp_input_dim, configs)

        elif configs.projector.projector_type == "MHACustomCNN":
            self.mlp = EsmClassificationHeadMHACustomCNN(mlp_input_dim, configs)

        # self.device = device
        self.configs = configs

    def forward(self, x, task_ids):

        if self.configs.encoder.model_name.startswith("esmc"):
            protein = ESMProtein(sequence=x[0])
            protein_tensor = self.esm2.encode(protein)

            logits_output = self.esm2.logits(
                protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )
            features = logits_output.embeddings
            # print(logits_output.logits, logits_output.embeddings)
        else:

            features = self.esm2(x['input_ids'].to(device), repr_layers=[self.esm2.num_layers], task_ids=task_ids,
                                 configs=self.configs)['representations'][self.esm2.num_layers]

        if self.configs.encoder.prompt.if_pass_to_MHA:
            # c = self.mlp(torch.concat([features[:, 0:500, :],features[:, 501:-1, :]],dim=1))
            c = self.mlp(torch.concat([features[:, 0:500, :], features[:, 501:-1, :]], dim=1))
        else:
            c = self.mlp(features[:, 1:-1, :])
        return c


def get_nb_trainable_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def print_trainable_parameters(model, logging):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params, all_param = get_nb_trainable_parameters(model)
    logging.info(
        f"trainable params: {trainable_params: ,} || all params: {all_param: ,} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_models(configs, logging):
    """
    Prepare the encoder model.

    Args:
        configs: A python box object containing the configuration options.
        logging: The logging object.

    Returns:
        The encoder model.
    """
    # Prepare the encoder.
    encoder = Encoder(logging=logging, configs=configs)
    print_trainable_parameters(encoder, logging)
    logging.info('encoder parameters: ' + str(sum(p.numel() for p in encoder.parameters())))

    return encoder


def prepare_models_merge(configs, logging):
    """
    Prepare the encoder model.

    Args:
        configs: A python box object containing the configuration options.
        logging: The logging object.

    Returns:
        The encoder model.
    """
    # Prepare the encoder.
    encoder = Encoder_merge(logging=logging, configs=configs)
    print_trainable_parameters(encoder, logging)
    logging.info('encoder parameters: ' + str(sum(p.numel() for p in encoder.parameters())))

    return encoder


def prepare_models_secondary_structure_ptm(configs, logging):
    """
    Prepare the encoder model.

    Args:
        configs: A python box object containing the configuration options.
        logging: The logging object.

    Returns:
        The encoder model.
    """
    # Prepare the encoder.
    encoder = EncoderSSPTM(logging=logging, configs=configs)
    print_trainable_parameters(encoder, logging)
    logging.info('encoder parameters: ' + str(sum(p.numel() for p in encoder.parameters())))

    return encoder


if __name__ == '__main__':
    # For test model and its modules
    config_path = './config.yaml'
    with open(config_path) as file:
        configs_dict = yaml.full_load(file)

    configs_file = load_configs(configs_dict)

    dummy_logging = get_dummy_logging()

    encoder_model = prepare_models(configs_file, dummy_logging)
    input_tensor = torch.randint(high=30, low=0, size=(2, 1024), dtype=torch.int64)

    sample = {'input_ids': input_tensor, 'attention_mask': torch.ones(input_tensor.shape)}
    output = encoder_model(sample)
    print(output.shape)
    print('done')

