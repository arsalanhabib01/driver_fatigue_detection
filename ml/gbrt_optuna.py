import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import copy
import bz2
from sklearn.metrics import roc_auc_score, precision_score, confusion_matrix, classification_report
import _pickle as cPickle
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
import warnings
import optuna

warnings.filterwarnings("ignore")

f = bz2.BZ2File("Experiments1_Datasets.pbz2", "rb")
data = cPickle.load(f)
f.close()


def toxcel(rep):
    nott=['weighted avg','accuracy','macro avg']
    l= [str(rep['accuracy']),
        str(rep['weighted avg']['f1-score']),
        str(rep['weighted avg']['precision']),
        str(rep['weighted avg']['recall']),'===']
    for k,v in rep.items():
        if k not in nott:
            l.extend([
                str(v['recall']*100),
                str(v['f1-score']),
                str(v['precision']),
                str(v['recall'])])
    return l

def old_code(data):
    labels = ['IT_B_Label', 'IT_M_Label', 'NST_B_Label', 'NST_M_Label']
    for indexLabel, label in enumerate(labels):
        for counter, dataPerRun in enumerate(data[indexLabel]):  # It will run 30 times
            print("\n**************** FOLD " + str(counter + 1) + " ************************\n")
            train, validation, test = dataPerRun[0], dataPerRun[1], dataPerRun[2]
            from lightgbm import LGBMClassifier

            X_train = train.iloc[:, :-4]
            X_validation = validation.iloc[:, :-4]
            X_test = test.iloc[:, :-4]

            y_train = train[label]
            y_validation = validation[label]
            y_test = test[label]

            # create dataset for lightgbm
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)
            lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

            # sc_1=StandardScaler()
            # sc_1.fit(df_1)

            if 'B' in label:
                # specify your configurations as a dict
                params = {
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'metric': {'binary_logloss'},
                    #   'n_estimators': 300,
                    #   'early_stopping_rounds': 5,
                    'max_depth': 5,
                    'num_leaves': 20,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': 0
                }

            else:
                params = {
                    'boosting_type': 'gbdt',
                    'objective': 'multiclass',
                    'metric': {'multi_logloss'},
                    'num_class': len(train[label].unique()),
                    #    'n_estimators': 300,
                    #    'early_stopping_rounds': 5,
                    'max_depth': 5,
                    'num_leaves': 20,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1
                }

            print('Starting training...')
            # train
            gbm = lgb.train(params,
                            lgb_train,
                            num_boost_round=20,
                            valid_sets=lgb_valid,
                            verbose_eval = False,
                            callbacks=[lgb.early_stopping(stopping_rounds=5)])

            print('Saving model...')
            # save model to file
            gbm.save_model('results/'+label + '_' + str(counter + 1) + '_model.txt')

            print('Starting predicting...')
            # predict
            y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

            if 'B' in label:
                # rounding the values
                y_pred = y_pred.round(0)
                # converting from float to integer
                y_pred = y_pred.astype(int)
                # roc_auc_score metric

                print('auc', roc_auc_score(y_pred, y_test))
            else:
                y_pred = [np.argmax(line) for line in y_pred]
                # using precision score for error metrics
                print('prec', precision_score(y_pred, y_test, average=None).mean())

            cr = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            print(cr)
            print(cm)

def report_average(reports):
    mean_dict = dict()
    for label in reports[0].keys():
        dictionary = dict()

        if label in 'accuracy':
            mean_dict[label] = sum(d[label] for d in reports) / len(reports)
            continue

        for key in reports[0][label].keys():
            dictionary[key] = sum(d[label][key] for d in reports) / len(reports)
        mean_dict[label] = dictionary

    return mean_dict

def objective_bin(trial):
    param_binary = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'feature_pre_filter': False,
        'verbose': -1,
        'seed': 42,
    }
    lgbcv = lgb.train(param_binary,
                      lgb_train,
                      valid_sets=lgb_valid,
                      early_stopping_rounds=15,
                      verbose_eval=False)
    score = lgbcv.best_score["valid_0"][param_binary['metric']]
    return score

def objective_mul(trial):
    param_multiclass = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'learning_rate': trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'feature_pre_filter': False,
        'seed': 42,
        'verbose': -1
    }
    param_multiclass['num_class'] = len(train[label].unique())

    lgbcv = lgb.train(param_multiclass,
                      lgb_train,
                      valid_sets=lgb_valid,
                      early_stopping_rounds=15,
                      verbose_eval=False)
    score = lgbcv.best_score["valid_0"][param_multiclass['metric']]
    return score


labels = ['IT_B_Label', 'IT_M_Label', 'NST_B_Label', 'NST_M_Label']
row = ['GBT']
for indexLabel, label in enumerate(labels):
    reports = []
    for counter, dataPerRun in enumerate(data[indexLabel]):  # It will run 30 times
        print("\n**************** FOLD " + str(counter + 1) + " ************************\n")
        train, validation, test = dataPerRun[0], dataPerRun[1], dataPerRun[2]
        X_train = train.iloc[:, :-4]
        X_validation = validation.iloc[:, :-4]
        X_test = test.iloc[:, :-4]
        y_train = train[label]
        y_validation = validation[label]
        y_test = test[label]
        # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train, params={'verbose': -1})
        lgb_valid = lgb.Dataset(X_test, y_test, params={'verbose': -1})
        lgb_test = lgb.Dataset(X_test, y_test, params={'verbose': -1})

        #optuna.logging.set_verbosity(optuna.logging.WARNING)

        # We search for another 4 hours (3600 s are an hours, so timeout=14400).
        # We could instead do e.g. n_trials=1000, to try 1000 hyperparameters chosen
        # by optuna or set neither timeout or n_trials so that we keep going until
        # the user interrupts ("Cancel run").
        study = optuna.create_study(direction='minimize')

        if 'B' in label:
            study.optimize(objective_bin, n_trials=100)
            best_params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    "verbosity": -1,
                    "boosting_type": "gbdt",
                    "seed": 42
            }
            best_params.update(study.best_params)
        else:
            study.optimize(objective_mul, n_trials=100)
            best_params = {
                "objective": "multiclass",
                'num_class': len(train[label].unique()),
                "metric": "multi_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "seed": 42,
                'verbose': -1,
            }
            best_params.update(study.best_params)

        lgbfit = lgb.train(best_params,
                           lgb_train,
                           verbose_eval=False,
                           num_boost_round=500)
        lgbfit.save_model('results/Experiment1_'+label + '_F' + str(counter + 1) + '_model.txt')
        y_pred = lgbfit.predict(X_test)
        if 'B' in label:
            # rounding the values
            y_pred = y_pred.round(0)
            # converting from float to integer
            y_pred = y_pred.astype(int)
            # roc_auc_score metric
            print('auc', roc_auc_score(y_pred, y_test))
        else:
            y_pred = [np.argmax(line) for line in y_pred]
            # using precision score for error metrics
            print('prec', precision_score(y_pred, y_test, average=None).mean())

        cr = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        print(cr)
        reports.append(cr)
        print(cm)
    print('****************************************************************************************************')
    print('****************************************************************************************************')
    print(label, 'all folds')
    avgrep = report_average(reports)
    row.extend(toxcel(avgrep))
    import csv
    with open("results/Experiment1_"+label+"_GBT_avg.csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(row)



'''
IT_B_Label all folds
{'0': {'precision': 0.43989096375834685, 'recall': 0.09480699247378621, 'f1-score': 0.15576327852862415, 'support': 3354.2}, '1': {'precision': 0.6629735911004487, 'recall': 0.936523562133103, 'f1-score': 0.7763448191981412, 'support': 6377.2}, 'accuracy': 0.6464023307953958, 'macro avg': {'precision': 0.5514322774293978, 'recall': 0.5156652773034446, 'f1-score': 0.4660540488633827, 'support': 9731.4}, 'weighted avg': {'precision': 0.5860818745843811, 'recall': 0.6464023307953958, 'f1-score': 0.5624439774934176, 'support': 9731.4}}

IT_M_Label all folds
{'0': {'precision': 0.1327950931477294, 'recall': 0.008220114381364206, 'f1-score': 0.015135173803520122, 'support': 535.4}, '1': {'precision': 0.251866034452798, 'recall': 0.03449205354367621, 'f1-score': 0.06035181425828494, 'support': 626.2}, '2': {'precision': 0.7014921339494097, 'recall': 0.19150364631387368, 'f1-score': 0.30040306682835954, 'support': 889.8}, '3': {'precision': 0.3181476931476932, 'recall': 0.04787924814885134, 'f1-score': 0.08140673046645844, 'support': 229.6}, '4': {'precision': 0.817482241282001, 'recall': 0.47974702571078565, 'f1-score': 0.6045294154061083, 'support': 893.4}, '5': {'precision': 0.5017619040749309, 'recall': 0.4205066767706649, 'f1-score': 0.4575093238096887, 'support': 3202.8}, '6': {'precision': 0.40565782519533294, 'recall': 0.7390139940120878, 'f1-score': 0.5237619199431173, 'support': 3354.2}, 'accuracy': 0.45847494055358357, 'macro avg': {'precision': 0.44702898932141366, 'recall': 0.27448039412590053, 'f1-score': 0.2918710635022196, 'support': 9731.4}, 'weighted avg': {'precision': 0.47517405886429576, 'recall': 0.45847494055358357, 'f1-score': 0.420709627768508, 'support': 9731.4}}

NST_B_Label all folds
{'0': {'precision': 0.8610027652423817, 'recall': 0.9959158594249391, 'f1-score': 0.9235572338271357, 'support': 7345.4}, '1': {'precision': 0.9757603883086734, 'recall': 0.5050293378038557, 'f1-score': 0.6655410310619834, 'support': 2386.0}, 'accuracy': 0.8755575051077014, 'macro avg': {'precision': 0.9183815767755276, 'recall': 0.7504725986143973, 'f1-score': 0.7945491324445595, 'support': 9731.4}, 'weighted avg': {'precision': 0.8891397422404463, 'recall': 0.8755575051077014, 'f1-score': 0.8602953641476816, 'support': 9731.4}}

NST_M_Label all folds
{'0': {'precision': 0.11210084033613446, 'recall': 0.0022415957595201563, 'f1-score': 0.004386496853670695, 'support': 535.4}, '1': {'precision': 0.4700028131384819, 'recall': 0.028744821682437284, 'f1-score': 0.05413551734519978, 'support': 626.2}, '2': {'precision': 0.9870354991376562, 'recall': 0.9547697532714015, 'f1-score': 0.9705374526011932, 'support': 141.6}, '3': {'precision': 0.45659300063392133, 'recall': 0.25856044723969257, 'f1-score': 0.32740887039362404, 'support': 53.4}, '4': {'precision': 0.9134174299207048, 'recall': 0.9389581356110936, 'f1-score': 0.9259791370195423, 'support': 435.8}, '5': {'precision': 0.9961061611861355, 'recall': 0.9457549071377065, 'f1-score': 0.9702714032706279, 'support': 593.6}, '6': {'precision': 0.8607505374718076, 'recall': 0.9967054900097987, 'f1-score': 0.9237520813994677, 'support': 7345.4}, 'accuracy': 0.8693508122291533, 'macro avg': {'precision': 0.6851437545464061, 'recall': 0.58939073581595, 'f1-score': 0.5966387084119036, 'support': 9731.4}, 'weighted avg': {'precision': 0.804647890508787, 'recall': 0.8693508122291533, 'f1-score': 0.8175572337958348, 'support': 9731.4}}
'''