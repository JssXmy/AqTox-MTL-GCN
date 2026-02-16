from sklearn.ensemble import RandomForestClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pandas as pd
from sklearn import metrics

parameters={}
# space of hyperopt parameters for Random Forest
space = {'max_depth': hp.choice('max_depth', list(range(3,20,1))),
         'min_samples_split': hp.choice('min_samples_split', list(range(2,11,1))),
         'min_samples_leaf': hp.choice('min_samples_leaf', list(range(1,5,1))),
         'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
         'n_estimators': hp.choice('n_estimators', list(range(100, 500, 50))),
         'bootstrap': hp.choice('bootstrap', [True, False]),
         'max_samples': hp.choice('max_samples', [0.7, 0.8, 0.9, 1.0]),
         }

task_list = ['FishCT','DMAT', 'FishAT',  'DMCT', 'AlgAT','AlgCT']
for rf_graph_feats_task in task_list:
    print('***************************************************************************************************')
    print(rf_graph_feats_task)
    print('***************************************************************************************************')
    args = {}
    training_set = pd.read_excel(rf_graph_feats_task+'_training_ECFP4.xlsx', index_col=None)
    valid_set = pd.read_excel(rf_graph_feats_task+'_valid_ECFP4.xlsx', index_col=None)
    test_set = pd.read_excel(rf_graph_feats_task+'_test_ECFP4.xlsx', index_col=None)
    x_colunms = [x for x in training_set.columns if x not in ['smiles', 'labels']]
    label_columns = ['labels']
    train_x = training_set[x_colunms]
    train_y = training_set[label_columns].values.ravel()
    valid_x = valid_set[x_colunms]
    valid_y = valid_set[label_columns].values.ravel()
    test_x = test_set[x_colunms]
    test_y = test_set[label_columns].values.ravel()


    def hyperopt_my_rf(parameter):
        model = RandomForestClassifier(
            max_depth=parameter['max_depth'],
            min_samples_split=parameter['min_samples_split'],
            min_samples_leaf=parameter['min_samples_leaf'],
            max_features=parameter['max_features'],
            n_estimators=parameter['n_estimators'],
            bootstrap=parameter['bootstrap'],
            max_samples=parameter['max_samples'] if parameter['bootstrap'] else None,
            random_state=2020,
            n_jobs=-1
        )
        model.fit(train_x, train_y)

        valid_prediction = model.predict_proba(valid_x)[:, 1]
        auc = metrics.roc_auc_score(valid_y, valid_prediction)
        return {'loss': -auc, 'status': STATUS_OK, 'model': model}


    # hyper parameter optimization
    trials = Trials()
    best = fmin(hyperopt_my_rf, space, algo=tpe.suggest, trials=trials, max_evals=50)
    print(best)

    # load the best model parameters
    args['max_depth'] = list(range(3,20,1))[best['max_depth']]
    args['min_samples_split'] = list(range(2,11,1))[best['min_samples_split']]
    args['min_samples_leaf'] = list(range(1,5,1))[best['min_samples_leaf']]
    args['max_features'] = ['sqrt', 'log2', None][best['max_features']]
    args['n_estimators'] = list(range(100, 500, 50))[best['n_estimators']]
    args['bootstrap'] = [True, False][best['bootstrap']]
    args['max_samples'] = [0.7, 0.8, 0.9, 1.0][best['max_samples']]

    # Create DataFrames for all metrics
    metric_names = ['Accuracy', 'Balanced_Accuracy', 'Precision', 'Recall', 'Specificity', 'F1_Score', 'ROC_AUC', 'PR_AUC']
    result_dfs = {}
    for metric_name in metric_names:
        result_dfs[metric_name] = pd.DataFrame()

    for i in range(10):
        model = RandomForestClassifier(
            max_depth=args['max_depth'],
            min_samples_split=args['min_samples_split'],
            min_samples_leaf=args['min_samples_leaf'],
            max_features=args['max_features'],
            n_estimators=args['n_estimators'],
            bootstrap=args['bootstrap'],
            max_samples=args['max_samples'] if args['bootstrap'] else None,
            random_state=2020 + i,
            n_jobs=-1
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
        result_dfs['Accuracy'].loc[i, rf_graph_feats_task] = accuracy
        result_dfs['Balanced_Accuracy'].loc[i, rf_graph_feats_task] = balanced_accuracy
        result_dfs['Precision'].loc[i, rf_graph_feats_task] = precision
        result_dfs['Recall'].loc[i, rf_graph_feats_task] = recall
        result_dfs['Specificity'].loc[i, rf_graph_feats_task] = specificity
        result_dfs['F1_Score'].loc[i, rf_graph_feats_task] = f1
        result_dfs['ROC_AUC'].loc[i, rf_graph_feats_task] = roc_auc
        result_dfs['PR_AUC'].loc[i, rf_graph_feats_task] = pr_auc

    # Save all metric results to separate CSV files
    for metric_name in metric_names:
        result_dfs[metric_name].to_csv(rf_graph_feats_task+'_ECFP4_rf_result_'+metric_name+'.csv', index=None)
    parameters[str(rf_graph_feats_task)]=args
filename = open('rf_ECFP4_parameters.txt', 'w')
for k,v in parameters.items():
    filename.write(k + ':' + str(v))
    filename.write('\n')
filename.close()

