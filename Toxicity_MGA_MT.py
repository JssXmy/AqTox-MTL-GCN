import numpy as np
from utils import build_dataset
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.MY_GNN import collate_molgraphs, EarlyStopping, run_a_train_epoch_heterogeneous, \
    set_random_seed, MGA, pos_weight, class_weight, run_an_eval_epoch_pih, \
    run_an_eval_epoch_heterogeneous_AROC, run_an_eval_epoch_heterogeneous_RBA, \
    run_an_eval_epoch_heterogeneous_all_metrics
from utils.MY_GNN import collate_predict, run_predict, load_pretrained
import os
import time
import pandas as pd
start = time.time()


# fix parameters of model
args = {}
args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
args['atom_data_field'] = 'atom'
args['bond_data_field'] = 'etype'
# 分类指标
args['classification_AROC'] = 'roc_auc'
args['classification_RBA']='RBA'

# 回归指标
args['regression_r2'] = 'r2'
args['regression_mse'] = 'mse'
args['regression_mae'] = 'mae'
args['regression_rmse'] = 'rmse'
# model parameter
args['num_epochs'] = 500
args['patience'] = 50
args['batch_size'] = 256
args['mode'] = 'higher'
args['in_feats'] = 40
args['rgcn_hidden_feats'] = [256, 128]
args['classifier_hidden_feats'] = 128
args['rgcn_drop_out'] = 0.4
args['drop_out'] = 0.3
args['lr'] = 3
args['weight_decay'] = 5
args['loop'] = True 

# task name (model name)
args['task_name'] = 'MTL-scr-256256128128'  # change
args['data_name'] = 'AquaTox'  # change
args['times'] = 10
# FishAT,DMCT,DMAT,AlgAT,FishCT,AlgCT,
# selected task, generate select task index, task class, and classification_num
args['select_task_list'] = ['FishCT','DMAT', 'FishAT',  'DMCT', 'AlgAT','AlgCT']  # change (excel list name) - removed pAlaGroErC50 as it's not in binary file
args['select_task_index'] = []
args['classification_num'] = 0
args['regression_num'] = 0
args['all_task_list'] = ['FishCT','DMAT', 'FishAT',  'DMCT', 'AlgAT','AlgCT']  # change (excel list name) - matches binary file structure
# generate select task index
for index, task in enumerate(args['all_task_list']):
    if task in args['select_task_list']:
        args['select_task_index'].append(index)

# generate classification_num
for task in args['select_task_list']:
    if task in ['FishCT','CruAT', 'FishAT',  'CruCT', 'AlgAT','AlgCT']:
        args['classification_num'] = args['classification_num'] + 1
    if task in ['logKow', 'pFishLC50', 'pFishEL_NOEC', 'pDMRepNOEC', 'pDMImbEC50', 'pAlaGroErC50']:
        args['regression_num'] = args['regression_num'] + 1

# generate classification_num
if args['classification_num'] != 0 and args['regression_num'] != 0:
    args['task_class'] = 'classification_regression'
if args['classification_num'] != 0 and args['regression_num'] == 0:
    args['task_class'] = 'classification'
if args['classification_num'] == 0 and args['regression_num'] != 0:
    args['task_class'] = 'regression'
print('Classification task:{}, Regression Task:{}'.format(args['classification_num'], args['regression_num']))

args['bin_path'] = 'data/' + args['data_name'] + '.bin'
args['group_path'] = 'data/' + args['data_name'] + '_group.csv'


# Create DataFrames for all metrics
metric_names = ['Accuracy', 'Balanced_Accuracy', 'Precision', 'Recall', 'Specificity', 'F1_Score', 'ROC_AUC', 'PR_AUC']
result_dfs = {}
for metric_name in metric_names:
    columns = args['select_task_list']+['group'] + args['select_task_list']+['group'] + args['select_task_list']+['group']
    result_dfs[metric_name] = pd.DataFrame(columns=columns)

all_times_train_result = []
all_times_val_result = []
all_times_test_result = []
for time_id in range(args['times']):
    set_random_seed(2020+time_id)
    one_time_train_result = []
    one_time_val_result = []
    one_time_test_result = []
    print('***************************************************************************************************')
    print('{}, {}/{} time'.format(args['task_name'], time_id+1, args['times']))
    print('***************************************************************************************************')
    train_set, val_set, test_set, task_number = build_dataset.load_graph_from_csv_bin_for_splited(
        bin_path=args['bin_path'],
        group_path=args['group_path'],
        select_task_index=args['select_task_index']
    )
    from torch.utils.data import ConcatDataset
    all_dataset = ConcatDataset([train_set, val_set, test_set])
    all_dataloader = DataLoader(dataset=all_dataset, batch_size=args['batch_size'], collate_fn=collate_molgraphs, shuffle=False)  
    print("Molecule graph generation is complete !")
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              collate_fn=collate_molgraphs)

    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            shuffle=True,
                            collate_fn=collate_molgraphs)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)
    pos_weight_np = pos_weight(train_set, classification_num=args['classification_num'])
    loss_criterion_c = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_np.to(args['device']))
    loss_criterion_r = torch.nn.MSELoss(reduction='none')

    model = MGA(in_feats=args['in_feats'], rgcn_hidden_feats=args['rgcn_hidden_feats'],
                n_tasks=task_number, rgcn_drop_out=args['rgcn_drop_out'],
                classifier_hidden_feats=args['classifier_hidden_feats'], dropout=args['drop_out'],
                loop=args['loop'])
    optimizer = Adam(model.parameters(), lr=10**-args['lr'], weight_decay=10**-args['weight_decay'])
    stopper = EarlyStopping(patience=args['patience'], task_name=args['task_name'], mode=args['mode'])
    model.to(args['device'])

    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch_heterogeneous(args, epoch, model, train_loader, loss_criterion_c, loss_criterion_r, optimizer)
        # run_a_train_epoch_heterogeneous(args, epoch, model, train_loader, loss_criterion_c, loss_criterion_r, optimizer, 
        #                                 loss_criterion_c_w = loss_criterion_c_w)

        # Validation and early stop
        validation_result = run_an_eval_epoch_heterogeneous_AROC(args, model, val_loader)
        val_score = np.mean(validation_result)
        early_stop = stopper.step(val_score, model)
        print('epoch {:d}/{:d}, validation {:.4f}, best validation {:.4f}'.format(
            epoch + 1, args['num_epochs'],
            val_score,  stopper.best_score)+' validation result:', validation_result)
        if early_stop:
            break
    stopper.load_checkpoint(model)

    # Evaluate all metrics
    train_metrics = run_an_eval_epoch_heterogeneous_all_metrics(args, model, train_loader)
    val_metrics = run_an_eval_epoch_heterogeneous_all_metrics(args, model, val_loader)
    test_metrics = run_an_eval_epoch_heterogeneous_all_metrics(args, model, test_loader)

    # Store results for each metric
    for metric_name in metric_names:
        result_row = train_metrics[metric_name] + ['train'] + val_metrics[metric_name] + ['valid'] + test_metrics[metric_name] + ['test']
        result_dfs[metric_name].loc[time_id] = result_row

    print('********************************{}, {}_times_result*******************************'.format(args['task_name'], time_id+1))
    print("train_result:")
    for metric_name in metric_names:
        print(f"  {metric_name}: {train_metrics[metric_name]}")
    print("val_result:")
    for metric_name in metric_names:
        print(f"  {metric_name}: {val_metrics[metric_name]}")
    print("test_result:")
    for metric_name in metric_names:
        print(f"  {metric_name}: {test_metrics[metric_name]}")
    run_an_eval_epoch_pih(args, model, all_dataloader, output_path='jieguo/'+args['task_name']+'prediction.csv')

# Save all metric results to separate CSV files
for metric_name in metric_names:
    result_dfs[metric_name].to_csv('result/'+args['task_name']+'_result_'+metric_name+'.csv', index=None)

elapsed = (time.time() - start)
m, s = divmod(elapsed, 60)
h, m = divmod(m, 60)
print("Time used:", "{:d}:{:d}:{:d}".format(int(h), int(m), int(s)))











