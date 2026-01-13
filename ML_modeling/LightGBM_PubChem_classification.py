import lightgbm as lgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pandas as pd
from sklearn import metrics

parameters={}
# space of hyperopt parameters for LightGBM
space = {'max_depth': hp.choice('max_depth', list(range(3,10,1))),
         'min_child_samples': hp.choice('min_child_samples', list(range(20,101,10))),
         'reg_lambda': hp.choice('reg_lambda', [1e-5, 1e-2, 0.1, 1]),
         'reg_alpha': hp.choice('reg_alpha', [1e-5, 1e-2, 0.1, 1]),
         'learning_rate': hp.choice('learning_rate', [0.01, 0.05, 0.001, 0.005]),
         'n_estimators': hp.choice('n_estimators', list(range(100, 300, 20))),
         'feature_fraction': hp.choice('feature_fraction', [i/100.0 for i in range(75,90,5)]),
         'bagging_fraction': hp.choice('bagging_fraction', [i/100.0 for i in range(75,90,5)]),
         'num_leaves': hp.choice('num_leaves', list(range(31,128,10))),
         }

task_list = ['FishLC50', 'FishEL_NOEC', 'DMRepNOEC', 'DMImbEC50', 'AlaGroErC50']
for lgb_graph_feats_task in task_list:
    print('***************************************************************************************************')
    print(lgb_graph_feats_task)
    print('***************************************************************************************************')
    args = {}
    training_set = pd.read_excel(lgb_graph_feats_task+'_training_PubChem.xlsx', index_col=None)
    valid_set = pd.read_excel(lgb_graph_feats_task+'_valid_PubChem.xlsx', index_col=None)
    test_set = pd.read_excel(lgb_graph_feats_task+'_test_PubChem.xlsx', index_col=None)
    x_colunms = [x for x in training_set.columns if x not in ['smiles', 'labels']]
    label_columns = ['labels']
    train_x = training_set[x_colunms]
    train_y = training_set[label_columns].values.ravel()
    valid_x = valid_set[x_colunms]
    valid_y = valid_set[label_columns].values.ravel()
    test_x = test_set[x_colunms]
    test_y = test_set[label_columns].values.ravel()


    def hyperopt_my_lgb(parameter):
        model = lgb.LGBMClassifier(
            learning_rate=parameter['learning_rate'],
            max_depth=parameter['max_depth'],
            min_child_samples=parameter['min_child_samples'],
            reg_alpha=parameter['reg_alpha'],
            reg_lambda=parameter['reg_lambda'],
            bagging_fraction=parameter['bagging_fraction'],
            feature_fraction=parameter['feature_fraction'],
            num_leaves=parameter['num_leaves'],
            n_estimators=parameter['n_estimators'],
            random_state=2020,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(train_x, train_y)

        valid_prediction = model.predict_proba(valid_x)[:, 1]
        auc = metrics.roc_auc_score(valid_y, valid_prediction)
        return {'loss': -auc, 'status': STATUS_OK, 'model': model}


    # hyper parameter optimization
    trials = Trials()
    best = fmin(hyperopt_my_lgb, space, algo=tpe.suggest, trials=trials, max_evals=50)
    print(best)

    # load the best model parameters
    args['max_depth'] = list(range(3,10,1))[best['max_depth']]
    args['min_child_samples'] = list(range(20,101,10))[best['min_child_samples']]
    args['reg_lambda'] = [1e-5, 1e-2, 0.1, 1][best['reg_lambda']]
    args['reg_alpha'] = [1e-5, 1e-2, 0.1, 1][best['reg_alpha']]
    args['learning_rate'] = [0.01, 0.05, 0.001, 0.005][best['learning_rate']]
    args['n_estimators'] = list(range(100, 300, 20))[best['n_estimators']]
    args['feature_fraction'] = [i / 100.0 for i in range(75, 90, 5)][best['feature_fraction']]
    args['bagging_fraction'] = [i / 100.0 for i in range(75, 90, 5)][best['bagging_fraction']]
    args['num_leaves'] = list(range(31,128,10))[best['num_leaves']]

    # Create DataFrames for all metrics
    metric_names = ['Accuracy', 'Balanced_Accuracy', 'Precision', 'Recall', 'Specificity', 'F1_Score', 'ROC_AUC', 'PR_AUC']
    result_dfs = {}
    for metric_name in metric_names:
        result_dfs[metric_name] = pd.DataFrame()

    for i in range(10):
        model = lgb.LGBMClassifier(
            learning_rate=args['learning_rate'],
            max_depth=args['max_depth'],
            min_child_samples=args['min_child_samples'],
            reg_alpha=args['reg_alpha'],
            reg_lambda=args['reg_lambda'],
            bagging_fraction=args['bagging_fraction'],
            feature_fraction=args['feature_fraction'],
            num_leaves=args['num_leaves'],
            n_estimators=args['n_estimators'],
            random_state=2020 + i,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(train_x, train_y.ravel())
        test_prediction = model.predict_proba(test_x)[:, 1]
        test_pred_labels = (test_prediction >= 0.5).astype(int)

        # Calculate all metrics
        accuracy = metrics.accuracy_score(test_y, test_pred_labels)
        balanced_accuracy = metrics.balanced_accuracy_score(test_y, test_pred_labels)
        precision = metrics.precision_score(test_y, test_pred_labels, zero_division=0)
        recall = metrics.recall_score(test_y, test_pred_labels, zero_division=0)
        tn, fp, fn, tp = metrics.confusion_matrix(test_y, test_pred_labels).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = metrics.f1_score(test_y, test_pred_labels, zero_division=0)
        roc_auc = metrics.roc_auc_score(test_y, test_prediction)
        precision_vals, recall_vals, _ = metrics.precision_recall_curve(test_y, test_prediction)
        pr_auc = metrics.auc(recall_vals, precision_vals)

        # Store results for each metric
        result_dfs['Accuracy'].loc[i, lgb_graph_feats_task] = accuracy
        result_dfs['Balanced_Accuracy'].loc[i, lgb_graph_feats_task] = balanced_accuracy
        result_dfs['Precision'].loc[i, lgb_graph_feats_task] = precision
        result_dfs['Recall'].loc[i, lgb_graph_feats_task] = recall
        result_dfs['Specificity'].loc[i, lgb_graph_feats_task] = specificity
        result_dfs['F1_Score'].loc[i, lgb_graph_feats_task] = f1
        result_dfs['ROC_AUC'].loc[i, lgb_graph_feats_task] = roc_auc
        result_dfs['PR_AUC'].loc[i, lgb_graph_feats_task] = pr_auc

    # Save all metric results to separate CSV files
    for metric_name in metric_names:
        result_dfs[metric_name].to_csv(lgb_graph_feats_task+'_PubChem_lgb_result_'+metric_name+'.csv', index=None)
    parameters[str(lgb_graph_feats_task)]=args
filename = open('lgb_PubChem_parameters.txt', 'w')
for k,v in parameters.items():
    filename.write(k + ':' + str(v))
    filename.write('\n')
filename.close()
