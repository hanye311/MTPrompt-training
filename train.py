import os
import numpy as np
import yaml
import argparse
import torch
import torchmetrics
from time import time, sleep
from tqdm import tqdm
from utils import load_configs, test_gpu_cuda, prepare_tensorboard, prepare_optimizer, save_checkpoint, \
    get_logging, load_checkpoints, prepare_saving_dir, focal_loss,EarlyStopping
from data import prepare_dataloaders_ptm
from model import prepare_models_secondary_structure_ptm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from focal_loss import FocalLoss
from torch.nn import functional as F
from collections import defaultdict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc,roc_auc_score,precision_score, recall_score,accuracy_score,f1_score,matthews_corrcoef
import json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pickle

def remove_label(tensor, output, label):
    mask = tensor != label
    return tensor[mask], output[mask]

def train(epoch, accelerator, dataloader, tools, global_step, tensorboard_log,configs,task_weights):
    # Initialize metrics
    accuracy = torchmetrics.Accuracy(task="binary")
    precision = torchmetrics.Precision(task="binary")
    recall = torchmetrics.Recall(task="binary")
    auc = torchmetrics.AUROC(task="binary")
    average_precision = torchmetrics.AveragePrecision(task="binary")
    f1_score = torchmetrics.F1Score(task="binary")
    mcc = torchmetrics.MatthewsCorrCoef(task="binary")

    accuracy.to(accelerator.device)
    precision.to(accelerator.device)
    recall.to(accelerator.device)
    auc.to(accelerator.device)
    average_precision.to(accelerator.device)
    f1_score.to(accelerator.device)
    mcc.to(accelerator.device)

    tools["optimizer"].zero_grad()

    epoch_loss = 0
    train_loss = 0
    counter = 0
    percent_done = 0

    progress_bar = tqdm(range(global_step, int(np.ceil(len(dataloader) / tools['accum_iter']))),
                        disable=not accelerator.is_local_main_process, leave=False)
    progress_bar.set_description("Steps")

    for i, data in enumerate(dataloader):
        with accelerator.accumulate(tools['net']):
            if configs.bam.model=='student':
                prot_id,sequences, labels, teacher_distill_output,masks, task_ids,_, _, indices = data

                outputs,_ = tools['net'](sequences,task_ids)

                log_probs = [torch.nn.functional.log_softmax(pred, dim=-1) for pred in outputs]

                # batch_labels = labels[masks]
                true_labels = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=configs.encoder.num_classes).float()

                teacher_labels = torch.nn.functional.softmax( teacher_distill_output / 1.0, dim=-1)

                if configs.bam.teacher_annealing:
                    percent_done=global_step/ (int(np.ceil(len(dataloader) / tools['accum_iter']))* 63)

                    if percent_done<=1:
                        new_labels = ((true_labels * percent_done) +
                                  (teacher_labels * (1 - percent_done)))
                    else:
                        new_labels = true_labels
                else:
                    percent_done=0
                    new_labels = ((true_labels * (1 - configs.bam.distill_weight)) +
                              (teacher_labels * configs.bam.distill_weight))

                batch_size = new_labels.size(0)
                total_loss = 0.0
                final_preds = []
                for i in range(batch_size):
                    example_task_id = task_ids[i].item()
                    logits = log_probs[example_task_id][i][masks[i]]  # Get the output for the correct task
                    final_preds.append(logits)
                    example_loss = -torch.sum(new_labels[i][masks[i]] * logits, dim=-1)
                    weighted_loss = example_loss * task_weights[example_task_id]
                    total_loss +=torch.mean(weighted_loss)
                loss =  total_loss / batch_size
                preds = torch.cat(final_preds, dim=0)


            else:
                prot_id, sequences, labels, masks, task_ids,_, _, indices = data

                outputs,_ = tools['net'](sequences, task_ids)

                batch_size =labels.size(0)
                total_loss = 0.0
                final_preds=[]
                for i in range(batch_size):
                    example_task_id = task_ids[i].item()
                    logits = outputs[example_task_id][i][masks[i]]  # Get the output for the correct task
                    final_preds.append(logits)


                    example_loss = tools['loss_function'](logits, labels[i][masks[i]].long().to(accelerator.device))
                    weighted_loss = example_loss * task_weights[example_task_id]
                    total_loss += torch.mean(weighted_loss)
                loss = total_loss / batch_size
                preds=torch.cat(final_preds,dim=0)

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(tools["train_batch_size"])).mean()
            train_loss += avg_loss.item() / tools['accum_iter']

            preds = F.softmax(preds, dim=-1)[:, 1]
            batch_labels = labels[masks]


            accuracy.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels).detach())
            precision.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels).detach())
            recall.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels).detach())
            auc.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels).detach())
            average_precision.update(accelerator.gather(preds).detach(),
                                     accelerator.gather(batch_labels.to(torch.int32)).detach())
            f1_score.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels).detach())
            mcc.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels).detach())

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(tools['net'].parameters(), tools['grad_clip'])

            tools['optimizer'].step()
            tools['scheduler'].step()
            tools['optimizer'].zero_grad()

        if accelerator.sync_gradients:
            if tensorboard_log:
                tools['train_writer'].add_scalar('step loss', train_loss, global_step)
                tools['train_writer'].add_scalar('learning rate', tools['optimizer'].param_groups[0]['lr'], global_step)

            progress_bar.update(1)
            global_step += 1
            accelerator.log({"train_loss": train_loss, 'lr': tools['optimizer'].param_groups[0]['lr']},
                            step=global_step)

            counter += 1
            epoch_loss += train_loss
            train_loss = 0

        logs = {"step_loss": loss.detach().item(),
                "lr": tools['optimizer'].param_groups[0]['lr']}
        progress_bar.set_postfix(**logs)

    train_loss = epoch_loss / counter
    epoch_acc = accuracy.compute().cpu().item()
    epoch_precision = precision.compute().cpu().item()
    epoch_recall = recall.compute().cpu().item()
    epoch_auc = auc.compute().cpu().item()
    epoch_average_precision = average_precision.compute().cpu().item()
    epoch_f1 = f1_score.compute().cpu().item()
    epoch_mcc = mcc.compute().cpu().item()

    accelerator.log({'train_precision': epoch_precision,
                     'train_recall': epoch_recall,
                     'train_auc': epoch_auc,
                     'train_average_precision': epoch_average_precision,
                     "train_f1": epoch_f1,
                     "train_mcc": epoch_mcc,
                     "train_acc": epoch_acc}, step=epoch)
    if tensorboard_log:
        tools['train_writer'].add_scalar('loss', train_loss, epoch)
        tools['train_writer'].add_scalar('accuracy', epoch_acc, epoch)
        tools['train_writer'].add_scalar('precision', epoch_precision, epoch)
        tools['train_writer'].add_scalar('recall', epoch_recall, epoch)
        tools['train_writer'].add_scalar('auc', epoch_auc, epoch)
        tools['train_writer'].add_scalar('average_precision', epoch_average_precision, epoch)
        tools['train_writer'].add_scalar('f1', epoch_f1, epoch)
        tools['train_writer'].add_scalar('mcc', epoch_mcc, epoch)

    # Reset metrics at the end of epoch
    accuracy.reset()
    precision.reset()
    recall.reset()
    auc.reset()
    average_precision.reset()
    f1_score.reset()
    mcc.reset()

    return train_loss, epoch_acc, epoch_precision, epoch_recall, epoch_auc, epoch_f1, epoch_mcc, epoch_average_precision,percent_done,global_step


def valid(epoch, accelerator, dataloader, tools, tensorboard_log,configs,task_weights):


    # Initialize metrics
    accuracy = torchmetrics.Accuracy(task="binary")
    precision = torchmetrics.Precision(task="binary")
    recall = torchmetrics.Recall(task="binary")
    auc = torchmetrics.AUROC(task="binary")
    average_precision = torchmetrics.AveragePrecision(task="binary")
    positive_f1_score = torchmetrics.F1Score(task="binary")
    mcc = torchmetrics.MatthewsCorrCoef(task="binary")

    accuracy.to(accelerator.device)
    precision.to(accelerator.device)
    recall.to(accelerator.device)
    auc.to(accelerator.device)
    average_precision.to(accelerator.device)
    positive_f1_score.to(accelerator.device)
    mcc.to(accelerator.device)

    counter = 0

    progress_bar = tqdm(range(len(dataloader)),
                        disable=not accelerator.is_local_main_process, leave=False)
    progress_bar.set_description("Steps")

    valid_loss = 0
    for i, data in enumerate(dataloader):
        if configs.bam.model=="student":
            prot_id, sequences, labels, teacher_distill_output, masks, task_ids,_, _, indices= data
        else:
            prot_id,sequences, labels, masks, task_ids,_, _, indices = data

        with torch.inference_mode():
            outputs,_ = tools['net'](sequences, task_ids)
            batch_size =labels.size(0)
            total_loss = 0.0
            final_preds=[]
            for i in range(batch_size):
                example_task_id = task_ids[i].item()
                logits = outputs[example_task_id][i][masks[i]]  # Get the output for the correct task
                final_preds.append(logits)
                example_loss = tools['loss_function'](logits, labels[i][masks[i]].long().to(accelerator.device))
                weighted_loss = example_loss * task_weights[example_task_id]
                total_loss += torch.mean(weighted_loss)
            loss = total_loss / batch_size
            preds=torch.cat(final_preds,dim=0)
            preds = F.softmax(preds, dim=-1)[:, 1]
            batch_labels = labels[masks]

            accuracy.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels.to(accelerator.device)).detach())
            precision.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels.to(accelerator.device)).detach())
            recall.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels.to(accelerator.device)).detach())
            auc.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels.to(accelerator.device)).detach())
            average_precision.update(accelerator.gather(preds).detach(),
                                     accelerator.gather(batch_labels.to(torch.int32).to(accelerator.device)).detach())
            positive_f1_score.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels.to(accelerator.device)).detach())
            mcc.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels.to(accelerator.device)).detach())

        counter += 1
        valid_loss += loss.data.item()

        progress_bar.update(1)
        logs = {"step_loss": loss.detach().item(),
                "lr": tools['optimizer'].param_groups[0]['lr']}

        progress_bar.set_postfix(**logs)

    valid_loss = valid_loss / counter

    epoch_acc = accuracy.compute().cpu().item()
    epoch_precision = precision.compute().cpu().item()
    epoch_recall = recall.compute().cpu().item()
    epoch_auc = auc.compute().cpu().item()
    epoch_average_precision = average_precision.compute().cpu().item()
    epoch_positive_f1 = positive_f1_score.compute().cpu().item()
    epoch_mcc = mcc.compute().cpu()

    accelerator.log({
        "valid_f1": epoch_positive_f1,
        "valid_precision": epoch_precision,
        "valid_recall": epoch_recall,
        "valid_auc": epoch_auc,
        "valid_average_precision": epoch_average_precision,
        "valid_acc": epoch_acc,
        "valid_mcc": epoch_mcc,
    },
        step=epoch)
    if tensorboard_log:
        tools['valid_writer'].add_scalar('loss', valid_loss, epoch)
        tools['valid_writer'].add_scalar('accuracy', epoch_acc, epoch)
        tools['valid_writer'].add_scalar('precision', epoch_precision, epoch)
        tools['valid_writer'].add_scalar('recall', epoch_recall, epoch)
        tools['valid_writer'].add_scalar('auc', epoch_auc, epoch)
        tools['valid_writer'].add_scalar('average_precision', epoch_average_precision, epoch)
        tools['valid_writer'].add_scalar('f1', epoch_positive_f1, epoch)
        tools['valid_writer'].add_scalar('mcc', epoch_mcc, epoch)

    # Reset metrics at the end of epoch
    accuracy.reset()
    precision.reset()
    recall.reset()
    auc.reset()
    average_precision.reset()
    positive_f1_score.reset()
    mcc.reset()

    return valid_loss, epoch_acc, epoch_precision, epoch_recall, epoch_auc, epoch_positive_f1, epoch_mcc, epoch_average_precision

def predict_distill(epoch, accelerator, dataloader, tools,tensorboard_log,configs):



    progress_bar = tqdm(range(len(dataloader)),
                        disable=not accelerator.is_local_main_process, leave=False)
    progress_bar.set_description("Steps")


    logits = {}
    for i, data in enumerate(dataloader):
        if configs.bam.model == "student":
            prot_id, sequences, labels, teacher_distill_output, masks, task_ids,_, _,indices = data
        else:
            prot_id, sequences, labels, masks, task_ids,_, _,indices = data

        with torch.inference_mode():
            outputs,_ = tools['net'](sequences, task_ids)
            batch_size = labels.size(0)

            for i in range(batch_size):
                example_task_id = task_ids[i].item()
                logits[prot_id[i]] = outputs[example_task_id][i]  # Get the output for the correct task

    return logits

def predict(epoch, accelerator, dataloader, tools, tensorboard_log,configs,task_weights):


    # Initialize metrics
    accuracy = torchmetrics.Accuracy(task="binary")
    precision = torchmetrics.Precision(task="binary")
    recall = torchmetrics.Recall(task="binary")
    auc = torchmetrics.AUROC(task="binary")
    average_precision = torchmetrics.AveragePrecision(task="binary")
    positive_f1_score = torchmetrics.F1Score(task="binary")
    mcc = torchmetrics.MatthewsCorrCoef(task="binary")

    accuracy.to(accelerator.device)
    precision.to(accelerator.device)
    recall.to(accelerator.device)
    auc.to(accelerator.device)
    average_precision.to(accelerator.device)
    positive_f1_score.to(accelerator.device)
    mcc.to(accelerator.device)

    counter = 0

    progress_bar = tqdm(range(len(dataloader)),
                        disable=not accelerator.is_local_main_process, leave=False)
    progress_bar.set_description("Steps")

    valid_loss = 0
    prediction_results=[]
    label_results=[]
    prot_id_results=[]
    positive_tensor_list=[]
    negative_tensor_list=[]
    for i, data in enumerate(dataloader):
        if configs.bam.model=="student":
            prot_id, sequences, labels, _, masks, task_ids, positive_masks,negative_masks,indices= data
        else:
            prot_id,sequences, labels, masks, task_ids, positive_masks,negative_masks,indices = data

        with torch.inference_mode():
            outputs,x_contact = tools['net'](sequences, task_ids)   # # x_contact,batch, length, out_channels
            positive_tensor_list.append(x_contact[positive_masks.reshape(-1)])
            negative_tensor_list.append(x_contact[negative_masks.reshape(-1)])
            batch_size =labels.size(0)
            total_loss = 0.0
            final_preds=[]
            for i in range(batch_size):
                example_task_id = task_ids[i].item()
                logits = outputs[example_task_id][i][masks[i]]  # Get the output for the correct task
                final_preds.append(logits)
                example_loss = tools['loss_function'](logits, labels[i][masks[i]].long().to(accelerator.device))
                weighted_loss = example_loss * task_weights[example_task_id]
                total_loss += torch.mean(weighted_loss)
            loss = total_loss / batch_size
            preds=torch.cat(final_preds,dim=0)
            preds = F.softmax(preds, dim=-1)[:, 1]
            batch_labels = labels[masks]
            prediction_results.extend(preds.tolist())
            label_results.extend(batch_labels.tolist())
            numbers_of_sequences=[final_preds[i].shape[0] for i in range(len(final_preds))]
            k=0
            for prot in list(prot_id):
                for i in range(numbers_of_sequences[k]) :
                    prot_id_results.append(prot)
                k+=1

            accuracy.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels.to(accelerator.device)).detach())
            precision.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels.to(accelerator.device)).detach())
            recall.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels.to(accelerator.device)).detach())
            auc.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels.to(accelerator.device)).detach())
            average_precision.update(accelerator.gather(preds).detach(),
                                     accelerator.gather(batch_labels.to(torch.int32).to(accelerator.device)).detach())
            positive_f1_score.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels.to(accelerator.device)).detach())
            mcc.update(accelerator.gather(preds).detach(), accelerator.gather(batch_labels.to(accelerator.device)).detach())

        counter += 1
        valid_loss += loss.data.item()

        progress_bar.update(1)
        logs = {"step_loss": loss.detach().item(),
                "lr": tools['optimizer'].param_groups[0]['lr']}

        progress_bar.set_postfix(**logs)

    valid_loss = valid_loss / counter

    epoch_acc = accuracy.compute().cpu().item()
    epoch_precision = precision.compute().cpu().item()
    epoch_recall = recall.compute().cpu().item()
    epoch_auc = auc.compute().cpu().item()
    epoch_average_precision = average_precision.compute().cpu().item()
    epoch_positive_f1 = positive_f1_score.compute().cpu().item()
    epoch_mcc = mcc.compute().cpu()

    accelerator.log({
        "valid_f1": epoch_positive_f1,
        "valid_precision": epoch_precision,
        "valid_recall": epoch_recall,
        "valid_auc": epoch_auc,
        "valid_average_precision": epoch_average_precision,
        "valid_acc": epoch_acc,
        "valid_mcc": epoch_mcc,
    },
        step=epoch)
    if tensorboard_log:
        tools['valid_writer'].add_scalar('loss', valid_loss, epoch)
        tools['valid_writer'].add_scalar('accuracy', epoch_acc, epoch)
        tools['valid_writer'].add_scalar('precision', epoch_precision, epoch)
        tools['valid_writer'].add_scalar('recall', epoch_recall, epoch)
        tools['valid_writer'].add_scalar('auc', epoch_auc, epoch)
        tools['valid_writer'].add_scalar('average_precision', epoch_average_precision, epoch)
        tools['valid_writer'].add_scalar('f1', epoch_positive_f1, epoch)
        tools['valid_writer'].add_scalar('mcc', epoch_mcc, epoch)

    # Reset metrics at the end of epoch
    accuracy.reset()
    precision.reset()
    recall.reset()
    auc.reset()
    average_precision.reset()
    positive_f1_score.reset()
    mcc.reset()

    positive_tensors=torch.cat(positive_tensor_list,dim=0)
    negative_tensors=torch.cat(negative_tensor_list,dim=0)

    return (valid_loss, epoch_acc, epoch_precision, epoch_recall, epoch_auc, epoch_positive_f1,
            epoch_mcc, epoch_average_precision,prediction_results,label_results,prot_id_results,positive_tensors,negative_tensors)


def main(args, dict_config, config_file_path):

    configs = load_configs(dict_config, args)
    # overwrite result_path and result_path if set through command

    if type(configs.fix_seed) == int:
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    torch.cuda.empty_cache()
    if device.type!='cpu':
        test_gpu_cuda()

    result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)

    logging = get_logging(result_path)

    if device.type=='cpu':
        accelerator = Accelerator(
            mixed_precision=configs.train_settings.mixed_precision,
            # split_batches=True,
            gradient_accumulation_steps=configs.train_settings.grad_accumulation,
            dispatch_batches=False,
            cpu=True
        )
    else:
        accelerator = Accelerator(
            mixed_precision=configs.train_settings.mixed_precision,
            # split_batches=True,
            gradient_accumulation_steps=configs.train_settings.grad_accumulation,
            dispatch_batches=False
        )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("accelerator_tracker", config=None)


    dataloaders_dict,task_weights= prepare_dataloaders_ptm(configs,logging)
    logging.info('preparing dataloaders are done')

    net = prepare_models_secondary_structure_ptm(configs, logging)
    logging.info('preparing model is done')

    if configs.bam.model=="student":
        if configs.bam.teacher_annealing==False:
            logging.info(f'Now is student mode, the distill value is : {configs.bam.distill_weight}.')
        else:
            logging.info(f'Now is student mode, teacher_annealing_epoch is 63.')
    elif configs.bam.model=="teacher":
        logging.info('Now is teacher mode.')

    optimizer, scheduler = prepare_optimizer(net, configs, len(dataloaders_dict["train"]), logging)
    logging.info('preparing optimizer is done')

    net, start_epoch = load_checkpoints(configs, optimizer, scheduler, logging, net)


    dataloaders_dict["train"], dataloaders_dict["valid"], dataloaders_dict["test"] = accelerator.prepare(
        dataloaders_dict["train"],
        dataloaders_dict["valid"],
        dataloaders_dict["test"]
    )

    net, optimizer, scheduler = accelerator.prepare(
        net, optimizer, scheduler
    )

    # initialize tensorboards
    train_writer, valid_writer = prepare_tensorboard(result_path)

    # prepare loss function
    if configs.train_settings.loss == 'crossentropy':
        criterion = torch.nn.CrossEntropyLoss(
            reduction="none",
            # label_smoothing=0.1
            weight=torch.tensor([1.0,3.0]).to(accelerator.device) #todo
            # ignore_index=dataloaders_dict["train"].encoder_tokenizer.tokens_dict['<pad>'],
        )
    elif configs.train_settings.loss == 'focal':
        alpha_value = torch.tensor([0.5,1])
        criterion = FocalLoss(
            alpha=alpha_value,
            gamma=2.0,
            reduction='none')
        criterion = accelerator.prepare(criterion)
    else:
        print('wrong loss!')
        exit()

    tools = {
        'net': net,
        'train_device': configs.train_settings.device,
        'valid_device': configs.valid_settings.device,
        'train_batch_size': configs.train_settings.batch_size,
        'valid_batch_size': configs.valid_settings.batch_size,
        'optimizer': optimizer,
        'mixed_precision': configs.train_settings.mixed_precision,
        'train_writer': train_writer,
        'valid_writer': valid_writer,
        'accum_iter': configs.train_settings.grad_accumulation,
        'loss_function': criterion,
        'grad_clip': configs.optimizer.grad_clip_norm,
        'checkpoints_every': configs.checkpoints_every,
        'scheduler': scheduler,
        'result_path': result_path,
        'checkpoint_path': checkpoint_path,
        'logging': logging,
        'num_classes': configs.encoder.num_classes
    }

    logging.info(f'number of train steps per epoch: {np.ceil(len(dataloaders_dict["train"]) / tools["accum_iter"])}')
    logging.info(f'number of valid steps per epoch: {len(dataloaders_dict["valid"])}')
    logging.info(f'number of test steps per epoch: {len(dataloaders_dict["test"])}')

    if configs.projector.if_frozen == True:
        # 3. Freeze CNN layers after loading the checkpoint
        for param in net.mlp.conv_layers.parameters():
            param.requires_grad = False
        for param in net.mlp.conv_layers2.parameters():
            param.requires_grad = False
        for param in net.mlp.fc_multiclass.linear_hidden.parameters():
            param.requires_grad = False



    # Training loop
    early_stopping = EarlyStopping(patience=20,metric=configs.save_checkpoint_metric)
    # best_val_loss = float('inf')
    best_valid_acc = 0
    best_valid_positive_f1 = 0
    best_valid_auc=0
    best_valid_loss = float('inf')
    best_valid_rpc = 0
    global_step = 0
    for epoch in range(start_epoch, configs.train_settings.num_epochs + 1):
        tools['epoch'] = epoch
        start_time = time()
        train_loss, train_acc, train_precision, train_recall, train_auc, train_f1, train_mcc, train_ap,percent_done,global_step = train(
            epoch, accelerator,
            dataloaders_dict["train"], tools, global_step,
            configs.tensorboard_log,
            configs,
            task_weights
        )
        end_time = time()
        if accelerator.is_main_process:
            logging.info(f'epoch {epoch} - time {np.round(end_time - start_time, 2)}s, '
                         f'train loss {np.round(train_loss, 4)}, train acc {np.round(train_acc, 4)}, '
                         f'train precision {np.round(train_precision, 4)}, '
                         f'train recall {np.round(train_recall, 4)}, '
                         f'train auc {np.round(train_auc, 4)}, '
                         f'train average precision {np.round(train_ap, 4)}, '
                         f'train f1 {np.round(train_f1, 4)}, '
                         f'train mcc {np.round(train_mcc, 4):.4f},'
                         f'percent_done {np.round(percent_done, 4):.4f},'
                         )

        if epoch % configs.valid_settings.do_every == 0 and epoch != 0:
            tools['net'].eval()
            sum_valid_f1=0
            sum_valid_acc=0
            sum_valid_auc=0
            sum_valid_loss=0
            sum_valid_rpc = 0
            for i,(task_name,dataloader) in enumerate(dataloaders_dict['valid'].items()):

                start_time = time()
                valid_loss, valid_acc, valid_precision, valid_recall, valid_auc, valid_positive_f1, valid_mcc, valid_ap = valid(
                    epoch, accelerator, dataloader,
                    tools, configs.tensorboard_log,configs,task_weights
                )
                end_time = time()
                sum_valid_f1+= valid_positive_f1
                sum_valid_acc+=valid_acc
                sum_valid_auc+=valid_auc
                sum_valid_loss += valid_loss
                sum_valid_rpc += valid_ap
                if accelerator.is_main_process:
                    logging.info(
                        f'\ntask name: {task_name},'
                        f'evaluation - time {np.round(end_time - start_time, 2)}s, '
                        f'valid loss {np.round(valid_loss, 4)}, valid acc {np.round(valid_acc, 4)}, '
                        f'valid precision {np.round(valid_precision, 4)}, '
                        f'valid recall {np.round(valid_recall, 4)}, '
                        f'valid auc {np.round(valid_auc, 4)}, '
                        f'valid average precision {np.round(valid_ap, 4)}, '
                        f'valid f1 {np.round(valid_positive_f1, 4)}, '
                        f'valid mcc {np.round(valid_mcc, 4):.4f}'
                    )
            avg_valid_positive_f1 = float(sum_valid_f1/(i+1))
            avg_valid_acc = float(sum_valid_acc/(i+1))
            avg_valid_auc=float(sum_valid_auc/(i+1))
            avg_valid_loss = float(sum_valid_loss / (i + 1))
            avg_valid_rpc = float(sum_valid_rpc / (i + 1))

            logging.info(f'\naverage valid acc: {np.round(avg_valid_acc, 4)}')
            logging.info(f'average valid positive f1: {np.round(avg_valid_positive_f1 , 4)}')
            logging.info(f'average valid auc: {np.round(avg_valid_auc, 4)}')
            logging.info(f'average valid loss: {np.round(avg_valid_loss, 4)}')
            logging.info(f'average valid rpc: {np.round(avg_valid_rpc, 4)}')

            if configs.save_checkpoint_metric=="loss":
                early_stopping(avg_valid_loss)
                if avg_valid_loss < best_valid_loss:
                    best_valid_acc = avg_valid_acc
                    best_valid_positive_f1 = avg_valid_positive_f1
                    best_valid_auc=avg_valid_auc
                    best_valid_loss = avg_valid_loss
                    # Set the path to save the model checkpoint.
                    model_path = os.path.join(tools['result_path'], 'checkpoints', f'best_model.pth')
                    accelerator.wait_for_everyone()
                    save_checkpoint(epoch, model_path, tools, accelerator)
                    print("Best checkpoint was saved in epoch " + str(epoch)+ "!")

            elif configs.save_checkpoint_metric=="auc":
                early_stopping(avg_valid_auc)

                if avg_valid_auc > best_valid_auc:
                    best_valid_acc = avg_valid_acc
                    best_valid_positive_f1 = avg_valid_positive_f1
                    best_valid_auc = avg_valid_auc
                    best_valid_loss = avg_valid_loss
                    # Set the path to save the model checkpoint.
                    model_path = os.path.join(tools['result_path'], 'checkpoints', f'best_model.pth')
                    accelerator.wait_for_everyone()
                    save_checkpoint(epoch, model_path, tools, accelerator)
                    print("Best checkpoint was saved in epoch " + str(epoch) + "!")

            elif configs.save_checkpoint_metric=="rpc":
                early_stopping(avg_valid_rpc)

                if avg_valid_rpc > best_valid_rpc:
                    best_valid_acc = avg_valid_acc
                    best_valid_positive_f1 = avg_valid_positive_f1
                    best_valid_auc = avg_valid_auc
                    best_valid_loss = avg_valid_loss
                    best_valid_rpc = avg_valid_rpc
                    # Set the path to save the model checkpoint.
                    model_path = os.path.join(tools['result_path'], 'checkpoints', f'best_model.pth')
                    accelerator.wait_for_everyone()
                    save_checkpoint(epoch, model_path, tools, accelerator)
                    print("Best checkpoint was saved in epoch " + str(epoch) + "!")

        if epoch % configs.checkpoints_every == 0:
            # Set the path to save the model checkpoint.
            model_path = os.path.join(tools['result_path'], 'checkpoints', f'checkpoint_{epoch}.pth')
            accelerator.wait_for_everyone()
            save_checkpoint(epoch, model_path, tools, accelerator)

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break


    if accelerator.is_main_process:
        logging.info(f'\nbest valid acc: {np.round(best_valid_acc, 4)}')
        logging.info(f'best valid positive f1: {np.round(best_valid_positive_f1, 4)}')
        logging.info(f'best valid auc: {np.round(best_valid_auc, 4)}')
        logging.info(f'best valid loss: {np.round(best_valid_loss, 4)}')

    train_writer.close()
    valid_writer.close()

    # pause 20 second to make sure the best validation checkpoint is ready on the disk
    sleep(20)

    model_path = os.path.join(tools['result_path'], 'checkpoints', 'best_model.pth')
    model_checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(model_checkpoint['model_state_dict'])

    for i, (task_name, dataloader) in enumerate(dataloaders_dict['test'].items()):
        tools['net'].eval()
        start_time = time()


        (test_loss, test_acc, test_precision, test_recall, test_auc, test_positive_f1, test_mcc, test_ap,
         prediction_results, label_results, prot_id_results,positive_tensors,negative_tensors) = predict(
            0, accelerator, dataloader, tools,
            tensorboard_log=False, configs=configs, task_weights=task_weights
        )

        end_time = time()



        if accelerator.is_main_process:
            logging.info(
                f'\ntask name: {task_name},'
                f'test - time {np.round(end_time - start_time, 2)}s, '
                f'test loss {np.round(test_loss, 4)}')
            logging.info(f'test acc: {np.round(test_acc, 4)}')
            logging.info(f'test precision: {np.round(test_precision, 4)}')
            logging.info(f'test recall: {np.round(test_recall, 4)}')
            logging.info(f'test auc: {np.round(test_auc, 4)}')
            logging.info(f'test average precision: {np.round(test_ap, 4)}')
            logging.info(f'test positive f1: {np.round(test_positive_f1, 4)}')
            logging.info(f'test mcc: {np.round(test_mcc, 4):.4f}')

        ##save prediction results into csv
        result_dic = {
            "prot_id": prot_id_results,
            "prediction": prediction_results,
            "label": label_results
        }
        df = pd.DataFrame(result_dic)
        df.to_csv(os.path.join(checkpoint_path,task_name + '_test_output.csv'),index=False)
        print( "The predictions of " + task_name + " has been saved!")


    ###########
    if configs.bam.model == 'teacher':
        if configs.bam.write_distill_outputs == True:
            for i, (task_name, dataloader) in enumerate(dataloaders_dict['valid'].items()):
                results = predict_distill(
                    0, accelerator, dataloader, tools,
                    tensorboard_log=False, configs=configs
                )

            output_pickle_path = os.path.join(configs.bam.teacher_distill_output_path, task_name, "valid")
            if not os.path.exists(output_pickle_path):
                # os.mkdir(output_pickle_path)
                os.makedirs(output_pickle_path, exist_ok=True)

            # Save tensor to a file using pickle
            with open(os.path.join(output_pickle_path, "predictions.pkl"), 'wb') as f:
                pickle.dump(results, f)
            print("Validation outputs are saved!")

            for i, (task_name, dataloader) in enumerate(dataloaders_dict['test'].items()):
                results = predict_distill(
                    0, accelerator, dataloader, tools,
                    tensorboard_log=False, configs=configs
                )

            output_pickle_path = os.path.join(configs.bam.teacher_distill_output_path, task_name, "test")
            if not os.path.exists(output_pickle_path):
                # os.mkdir(output_pickle_path)
                os.makedirs(output_pickle_path, exist_ok=True)

            # Save tensor to a file using pickle
            with open(os.path.join(output_pickle_path, "predictions.pkl"), 'wb') as f:
                pickle.dump(results, f)
            print("Testing outputs are saved!")

            results = predict_distill(
                0, accelerator,
                dataloaders_dict["train"], tools,
                tensorboard_log=False, configs=configs
            )

            output_pickle_path = os.path.join(configs.bam.teacher_distill_output_path, task_name, "train")
            if not os.path.exists(output_pickle_path):
                # os.mkdirs(output_pickle_path)
                os.makedirs(output_pickle_path, exist_ok=True)

            # Save tensor to a file using pickle
            with open(os.path.join(output_pickle_path, "predictions.pkl"), 'wb') as f:
                pickle.dump(results, f)
            print("Training outputs are saved!")

            ############

    accelerator.end_training()
    accelerator.free_memory()
    del tools, net, dataloaders_dict, accelerator, optimizer, scheduler
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a classification model using esm")
    parser.add_argument("--config_path", "-c", help="The location of config file",
                        default='configs_all/PTM_config_adapterH_prompt.yaml')
    parser.add_argument("--result_path", default=None,
                        help="result_path, if setted by command line, "
                             "overwrite the one in config.yaml, by default is None")
    parser.add_argument("--resume_path", default=None,
                        help="if set, overwrite the one in config.yaml, by default is None")
    parser.add_argument("--num_end_adapter_layers", default=None, help="num_end_adapter_layers")
    parser.add_argument("--module_type", default=None, help="module_type for adapterh")
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(args, config_file, config_path)
    print('done!')