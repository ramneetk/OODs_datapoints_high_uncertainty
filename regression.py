"""
Created on Sun Oct 21 2018
@author: Kimin Lee
"""
from __future__ import print_function
import numpy as np
import os
import lib_regression
import argparse
import yaml
import pickle
import csv

from sklearn.linear_model import LogisticRegressionCV
from sklearn import preprocessing

# parse command line arguments for the config file
def parse_args(): 
    parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/OOD_Regression_Mahalanobis_config.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc, flush=True)

    return config

def main():
    # parse command line arguments for the config file
    config = parse_args()

    # initial setup
    dataset_list = config['model_params']['dataset_list']
    score_list = config['exp_params']['score_list']
    
    # train and measure the performance of Mahalanobis detector
    list_best_results, list_best_results_index = [], []
    for dataset in dataset_list:
        #print('In-distribution: ', dataset)
        outf = config['logging_params']['outf']
        out_list = config['model_params']['out_dist_list']

        list_best_results_out, list_best_results_index_out = [], []

        num_train = config['model_params']['num_train']
        for out in out_list:
            #print('Out-of-distribution: ', out)
            if out == 'subset_cifar100':
                num_train = 200
            best_tnr, best_result, best_index = 0, 0, 0
            for score in score_list:
                # preprocessor = preprocessing.MinMaxScaler((-1,1))
                preprocessor = preprocessing.StandardScaler()
                total_X, total_Y = lib_regression.load_characteristics(score, dataset, out, outf)
                #print("Shape of total_X: ", total_X.shape)
                X_val, Y_val, X_test, Y_test = lib_regression.block_split(total_X, total_Y, out,config)
                #print("Shape of X_val {}, X_test {}".format(X_val.shape, X_test.shape))
                X_train = np.concatenate((X_val[:num_train//2], X_val[num_train:num_train+num_train//2]))
                # Learning preprocessing params on train features and also transforming them
                X_train = preprocessor.fit_transform(X_train)
                Y_train = np.concatenate((Y_val[:num_train//2], Y_val[num_train:num_train+num_train//2]))
                X_val_for_test = np.concatenate((X_val[num_train//2:num_train], X_val[num_train+num_train//2:num_train*2]))
                # Transforming validation data
                X_val_for_test = preprocessor.transform(X_val_for_test)
                Y_val_for_test = np.concatenate((Y_val[num_train//2:num_train], Y_val[num_train+num_train//2:num_train*2]))
                lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
                ##############################################################################
                #################### LOADING Reg and Pre models ##############################
                ##############################################################################
                #lr = pickle.load(open('FGSM_mahala_regressor_model.sav', 'rb'))
                #preprocessor = pickle.load(open('FGSM_mahala_preprocessor_model.sav', 'rb'))

                #y_pred = lr.predict_proba(X_train)[:, 1]
                #print('training mse: {:.4f}'.format(np.mean(y_pred - Y_train)))
                #y_pred = lr.predict_proba(X_val_for_test)[:, 1]
                #print('test mse: {:.4f}'.format(np.mean(y_pred - Y_val_for_test)))
                results, ood_threshold = lib_regression.detection_performance(lr, X_val_for_test, Y_val_for_test, outf)
                #print("RESULTS on validation data - ",results)
                if best_tnr <= results['TMP']['TNR']:
                    best_tnr = results['TMP']['TNR']
                    best_index = score
                    #Transforming test data
                    X_test = preprocessor.transform(X_test)
                    best_result, ood_threshold = lib_regression.detection_performance(lr, X_test, Y_test, outf)
                    ##############################################################################
                    ##################### SAVING Reg and Pre models ##############################
                    ##############################################################################
                    #print("Saving for K: ", score)
                    pickle.dump(lr, open('adv_svhn_regressor_model.sav', 'wb'))
                    pickle.dump(preprocessor, open('adv_svhn__preprocessor_model.sav', 'wb'))
                #print("********OOD threshold- ", ood_threshold)
            list_best_results_out.append(best_result)
            list_best_results_index_out.append(best_index)
        list_best_results.append(list_best_results_out)
        list_best_results_index.append(list_best_results_index_out)
        
    # print the results
    count_in = 0
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']

    for in_list in list_best_results:
        print('in_distribution: ' + dataset_list[count_in] + '==========')
        out_list = config['model_params']['out_dist_list']
        count_out = 0
        for results in in_list:
            print('out_distribution: '+ out_list[count_out])
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*results['TMP']['TNR']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
            print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')
            with open(config['logging_params']['output_file']+'_out-dist_'+out_list[count_out]+'.csv', 'a', newline='') as csvfile: 
                spamwriter = csv.writer(csvfile, delimiter=',') 
                spamwriter.writerow([config['logging_params']['exp_display_name'],
                                    '{val:6.2f}'.format(val=100.*results['TMP']['TNR']),
                                    '{val:6.2f}'.format(val=100.*results['TMP']['AUROC']),
                                    '{val:6.2f}'.format(val=100.*results['TMP']['DTACC']),
                                    '{val:6.2f}'.format(val=100.*results['TMP']['AUIN']),
                                    '{val:6.2f}'.format(val=100.*results['TMP']['AUOUT'])]) 
            print('Input noise: ' + list_best_results_index[count_in][count_out])
            print('')
            count_out += 1
        count_in += 1
    
    np.savez('ood_threshold.npz', ood_threshold = ood_threshold)   # assuming only 1 out_dist here

if __name__ == '__main__':
    main()
