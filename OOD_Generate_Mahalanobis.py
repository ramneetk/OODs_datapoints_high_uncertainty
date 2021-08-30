"""
Created on Sun Oct 21 2018
@author: Kimin Lee
"""
from __future__ import print_function
import argparse
import torch
import data_loader
import numpy as np
import calculate_log as callog
from models import *
import os
import lib_generation
import utils
import yaml
from torchvision import transforms
from torch.autograd import Variable

# parse command line arguments for the config file
def parse_args(): 
    parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/OOD_Generate_Mahalanobis_config.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc, flush=True)

    return config

def get_inputs_for_computing_regressor_feature(regressor_features, model, config, num_output, feature_list, train_loader, device):
    
    num_classes = config['exp_params']['num_classes']

    class_mean, class_precision, tied_precision, tied_mean, knn_search_list, labels = None, None, None, None, None, None
    pca_list = []
    
    for regressor_feature in regressor_features:
        if regressor_feature == 'mahalanobis_tied_mean_tied_cov':
            #print("get mean and precision for {}".format(regressor_feature))
            tied_mean, tied_precision = utils.calc_tied_mean_tied_precision(model, num_classes, feature_list, train_loader, device)

        if regressor_feature == 'mahalanobis_tied_cov':
            #print("get mean and precision for {}".format(regressor_feature))
            class_mean, tied_precision = utils.calc_class_mean_tied_precision(model, num_classes, feature_list, train_loader, device)
        
        # we need class_mean and class_precision for introducing noise in the input features acc to the Mahalanobis paper
        if regressor_feature == 'mahalanobis_class_cov' or regressor_feature == 'pca' and class_precision == None :
            #print("get mean and precision for {}".format(regressor_feature))
            class_mean, class_precision = utils.calc_class_mean_class_precision(model, num_classes, feature_list, train_loader, device)
        
        if regressor_feature == 'pca':
            #print("get pca_list for {}".format(regressor_feature))
            pca_list = lib_generation.get_pca(model, num_classes, feature_list, train_loader, device)
        
        if regressor_feature == 'knn_label_conformance':
            knn_args = config['exp_params']['knn_args']
            # set the KNNSearch class params according to the knn search library type
            if knn_args['knn_type'] =='datasketch':
                knn_type_args = {'algorithm':'datasketch', 'num_perm':128,'normalize':False,'create':False,'file_path':'datasketch.pkl'}
            elif knn_args['knn_type'] =='annoy':
                knn_type_args = {'algorithm':'annoy','num_trees':128,'normalize':False,'metric':'euclidean','create':True,'file_path':'annoy_{}.ann'.format(config['trainer_params']['gpu'])}
            elif knn_args['knn_type'] =='exact':
                knn_type_args = {'algorithm':'exact','normalize':False}
            elif knn_args['knn_type'] =='random':
                knn_type_args = {'algorithm': 'random','normalize':False}
            elif knn_args['knn_type'] =='falconn':
                knn_type_args = {'algorithm': 'falconn','number_bits':17,'nb_tables':200,'normalize':True}
            # elif knn_args['knn_type'] =='descent':
            #     knn_type_args = {'algorithm': 'descent','metric':'euclidean', 'normalize':False}
            else:
                raise Exception('Wrong KNN algorithm input')
                #print("Calling knn_label_conformance")
            knn_search_list, labels = utils.calc_knnSearch_and_labels(model,
                            num_classes,
                            feature_list,
                            train_loader,
                            device,
                            knn_type_args)

        
    return class_mean, class_precision, tied_mean, tied_precision, pca_list, knn_search_list, labels


def get_inputs_for_computing_knn_regressor_feature(regressor_feature, 
                                                   model, 
                                                   config, 
                                                   num_output, 
                                                   feature_list, 
                                                   train_loader, 
                                                   k, 
                                                   device,
                                                   class_precision = None,
                                                   tied_precision = None,
                                                   class_mean = None):

    num_classes = config['exp_params']['num_classes']

    #print("get knn_serch_list, mean and precision for {}".format(regressor_feature))

    knn_args = config['exp_params']['knn_args']
    if regressor_feature == 'knn_mahalanobis_class_cov':
        knn_cov_type = 'class_cov'
        # for adding noise
        if class_precision == None:
            class_mean, class_precision = utils.calc_class_mean_class_precision(model, num_classes, feature_list, train_loader, device)

    else:
        knn_cov_type = 'tied_cov'
        # for adding noise
        if tied_precision == None:
            class_mean, tied_precision = utils.calc_class_mean_tied_precision(model, num_classes, feature_list, train_loader, device)

    # set the KNNSearch class params according to the knn search library type
    if knn_args['knn_type'] =='datasketch':
        knn_type_args = {'algorithm':'datasketch', 'num_perm':128,'normalize':False,'create':False,'file_path':'datasketch.pkl'}
    elif knn_args['knn_type'] =='annoy':
        knn_type_args = {'algorithm':'annoy','num_trees':128,'normalize':False,'metric':'euclidean','create':True,'file_path':'annoy_{}.ann'.format(config['trainer_params']['gpu'])}
    elif knn_args['knn_type'] =='exact':
        knn_type_args = {'algorithm':'exact','normalize':False}
    elif knn_args['knn_type'] =='random':
        knn_type_args = {'algorithm': 'random','normalize':False}
    elif knn_args['knn_type'] =='falconn':
        knn_type_args = {'algorithm': 'falconn','number_bits':17,'nb_tables':200,'normalize':True}
    # elif knn_args['knn_type'] =='descent':
    #     knn_type_args = {'algorithm': 'descent','metric':'euclidean', 'normalize':False}
    else:
        raise Exception('Wrong KNN algorithm input')
    
    # feature_list for knn will have different dimension than the original feature_list
    if config['exp_params']['dataset'] == 'mnist':
        temp_x =  torch.rand(2,1,32,32).to(device) 
    elif config['exp_params']['dataset'] == 'toy_data':
        temp_x =  torch.rand(2,2).to(device)
    else :
        temp_x = torch.rand(2,3,32,32).to(device)            
    #temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    knn_feature_list = np.zeros(num_output)
    count = 0
    for out in temp_list:
        if config['exp_params']['knn_args']['keep_original']:
            knn_feature_list[count] = out.size(1)
        if config['exp_params']['knn_args']['keep_knn_mean']:
            knn_feature_list[count]+=out.size(1)
        if config['exp_params']['knn_args']['keep_knn_std']:
            knn_feature_list[count]+=out.size(1)
        count += 1

    #print("Calling calc_knn_mean_precision")
    knn_search_list, knn_mean, knn_precision = utils.calc_knn_mean_precision(model = model, 
                                                                            num_classes = num_classes,
                                                                            feature_list = knn_feature_list, 
                                                                            train_loader = train_loader,
                                                                            device = device, 
                                                                            cov_type = knn_cov_type, 
                                                                            knn_type_args = knn_type_args, 
                                                                            knn_args = knn_args,
                                                                            k = k)
    #print("Done calc_knn_mean_precision")


    return class_mean, class_precision, tied_precision, knn_search_list, knn_mean, knn_precision


def get_features_for_regressor(regressor_feature, model, config, test_loader, dataset, i, out_flag, device,
                               class_mean, class_precision, tied_mean, tied_precision, pca_list, knn_search_list, knn_mean, knn_precision, 
                               knn_conformance_search_list, conformance_labels,
                               k=0):

    if regressor_feature == 'mahalanobis_class_cov':

        #print("Getting scores using Mahalanobis class covoriance")
        scores = []
        for magnitude in config['exp_params']['noise_params']['m_list']:
            cur_score = lib_generation.get_Mahalanobis_score(regressor_feature, model, config, test_loader, out_flag, class_mean, class_precision,
                                                    class_mean, class_precision, i, magnitude, 
                                                    knn_search_list[i], device, knn=False, k=k)
            cur_score = np.array(cur_score)
            scores.append(cur_score.reshape(-1,1))
        return np.hstack(scores)

    elif regressor_feature == 'mahalanobis_tied_cov':

        #print("Getting scores using Mahalanobis tied covoriance [Mahalanobis Paper]")
        scores = []
        for magnitude in config['exp_params']['noise_params']['m_list']:
            cur_score = lib_generation.get_Mahalanobis_score(regressor_feature, model, config, test_loader, out_flag, class_mean, tied_precision,
                                                    class_mean, tied_precision, i, magnitude, 
                                                    knn_search_list[i], device, knn=False , k=k)
            cur_score = np.array(cur_score)            
            scores.append(cur_score.reshape(-1,1))
        return np.hstack(scores)
    
    elif regressor_feature == 'mahalanobis_tied_mean_tied_cov':

        #print("Getting scores using Mahalanobis tied covoriance")
        scores = []
        for magnitude in config['exp_params']['noise_params']['m_list']:
            cur_score = lib_generation.get_Mahalanobis_score(regressor_feature, model, config, test_loader, out_flag, tied_mean, tied_precision,
                                                    tied_mean, tied_precision, i, magnitude, 
                                                    knn_search_list[i], device, knn=False , k=k)
            cur_score = np.array(cur_score)            
            scores.append(cur_score.reshape(-1,1))
        return np.hstack(scores)
    
    elif regressor_feature == 'pca':

        #print("Getting scores using PCA")        
        scores = []
        for magnitude in config['exp_params']['noise_params']['m_list']:
            cur_score = lib_generation.get_pca_score(model, config, test_loader, out_flag,
                                                    class_mean, class_precision, i, magnitude, pca_list, device)
            cur_score = np.array(cur_score)
            scores.append(cur_score.reshape(-1,1))
        return np.hstack(scores)

    elif regressor_feature == 'knn_mahalanobis_class_cov':

        #print("Getting scores using Mahalanobis class covoriance on K-NNs")
        scores = []
        for magnitude in config['exp_params']['noise_params']['m_list']:
            cur_score = lib_generation.get_Mahalanobis_score(regressor_feature, model, config, test_loader, out_flag, class_mean, class_precision,
                                                    knn_mean, knn_precision, i, magnitude, 
                                                    knn_search_list[i], device, knn=True, k=k)
            cur_score = np.array(cur_score)
            scores.append(cur_score.reshape(-1,1))
        return np.hstack(scores)
        
    elif regressor_feature == 'knn_mahalanobis_tied_cov':

        #print("Getting scores using Mahalanobis tied covoriance on K-NNs")
        scores = []
        for magnitude in config['exp_params']['noise_params']['m_list']:
            cur_score = lib_generation.get_Mahalanobis_score(regressor_feature, model, config, test_loader, out_flag, class_mean, tied_precision,
                                                    knn_mean, knn_precision, i, magnitude, 
                                                    knn_search_list[i], device, knn=True, k=k)
            cur_score = np.array(cur_score)
            scores.append(cur_score.reshape(-1,1))
        return np.hstack(scores)

    elif regressor_feature == 'ODIN':
        scores = []
        for params in config['exp_params']['odin_args']['settings']:
            cur_score = lib_generation.get_posterior(model, config['model_params']['net_type'], test_loader, params[1], params[0], config['logging_params']['outf'], out_flag,device)
            scores.append(cur_score.reshape(-1,1))

        return np.hstack(scores)
    
    elif regressor_feature == 'knn_label_conformance':
        #print("Getting knn label conformance scores")
        k = config['exp_params']['knn_args']['k'][0]
        scores = []
        for magnitude in config['exp_params']['noise_params']['m_list']:
            cur_score = lib_generation.get_knn_conformance_score(model, config, test_loader, out_flag, i,
                                                    knn_conformance_search_list[i], conformance_labels, device, k=k)
            cur_score = np.array(cur_score)
            scores.append(cur_score.reshape(-1,1))
        return np.hstack(scores)

    # This part is not tested and most likely wrong!
    # elif regressor_feature == 'LID':
    #     # dumping code
    #     os.system("python ADV_Samples.py --dataset {} --net_type {} --adv_type {} --gpu {} --outf {} --model {} --ood_idx {} --num_oods {}".format(dataset,
    #     config['model_params']['net_type'], config['exp_params']['lid_args']['adv_type'], config['trainer_params']['gpu'], config['exp_params']['lid_args']['outf'], 
    #     config['exp_params']['dataset'], config['model_params']['out_idx'], config['model_params']['num_oods']))
    #     # scoring code
    #     base_path = config['exp_params']['lid_args']['outf'] + config['model_params']['net_type'] + '_' + dataset + '/'
    #     test_clean_data = torch.load(base_path + 'clean_data_%s_%s_%s.pth' % (config['model_params']['net_type'], dataset, config['exp_params']['lid_args']['adv_type']))
    #     test_adv_data = torch.load(base_path  + 'adv_data_%s_%s_%s.pth' % (config['model_params']['net_type'], dataset, config['exp_params']['lid_args']['adv_type']))
    #     test_noisy_data = torch.load(base_path  + 'noisy_data_%s_%s_%s.pth' % (config['model_params']['net_type'], dataset, config['exp_params']['lid_args']['adv_type']))
    #     test_label = torch.load(base_path + 'label_%s_%s_%s.pth' % (config['model_params']['net_type'], dataset, config['exp_params']['lid_args']['adv_type']))
    #     LID, LID_adv, LID_noisy = lib_generation.get_LID(model, test_clean_data, test_adv_data, test_noisy_data, test_label, i+1)
    #     LID_scores = np.hstack([np.vstack(s) for s in LID])
    #     print("LID_scores_shape:",LID_scores.shape)
    #     return LID_scores

    else:
        raise Exception("Wrong type of regressor feature")

def main(config, device, model, in_transform, train_loader, test_loader):
    
    # set information about feature extaction
    model.eval()
    if config['exp_params']['dataset'] == 'mnist':
        temp_x =  torch.rand(2,1,32,32).to(device) 
    elif config['exp_params']['dataset'] == 'toy_data':
        temp_x =  torch.rand(2,2).to(device)
    else :
        temp_x = torch.rand(2,3,32,32).to(device)

    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    #print("***temp_list.shape: ", temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
        
    #print("Feature list: ", feature_list)
    # m_list is the list for magnitutde of noise
    m_list = config['exp_params']['noise_params']['m_list']

    # calculate input required for getting features for regressor
    regressor_features = config['exp_params']['regressor_features']

    # get the list of number of nearest neighbors
    num_knns = config['exp_params']['knn_args']['k']

    # get input for all regressor features except for KNN
    class_mean, class_precision, tied_mean, tied_precision, pca_list, knn_conformance_search_list, conformance_labels = get_inputs_for_computing_regressor_feature(regressor_features, model, config, num_output, feature_list, train_loader, device)

    # get inputs for knn for all values of k
    knn_feature_flag = False
    only_knn_feature = True
    if 'knn_mahalanobis_class_cov' in regressor_features or 'knn_mahalanobis_tied_cov' in regressor_features:
        knn_feature_flag = True
        knn_feature_name = 'knn_mahalanobis_class_cov' if 'knn_mahalanobis_class_cov' in regressor_features else 'knn_mahalanobis_tied_cov'
        knn_inputs = {}
        for k in num_knns:
            knn_inputs[k] = get_inputs_for_computing_knn_regressor_feature(knn_feature_name, model, config, num_output, 
                            feature_list, train_loader, k, device, class_precision, tied_precision, class_mean)


    init_reg_in = True
    #print("For in-distribution: {}".format(config['exp_params']['dataset']))
    for regressor_feature in regressor_features:
        if  regressor_feature == 'knn_mahalanobis_class_cov' or regressor_feature == 'knn_mahalanobis_tied_cov':
            continue
        only_knn_feature = False

        # num_output is the number of layers
        for i in range(num_output):
            in_dist_input = get_features_for_regressor(regressor_feature, model, config, test_loader, config['exp_params']['dataset'], i, True, device,
                                                class_mean, class_precision, tied_mean, tied_precision, pca_list, knn_search_list=[None]*num_output, knn_mean=None, knn_precision=None,
                                                knn_conformance_search_list=knn_conformance_search_list, conformance_labels=conformance_labels, k=0)

            #print("in_dist_input shape- ", in_dist_input.shape)
            in_dist_input = np.asarray(in_dist_input, dtype=np.float32)

            #print("Mean score at layer {} for regression type {}: {}".format(i,regressor_feature,np.mean(in_dist_input)))

            if init_reg_in:
                regressor_in_dist_input = in_dist_input.reshape((in_dist_input.shape[0], -1))
                init_reg_in = False
            else:
                regressor_in_dist_input = np.concatenate((regressor_in_dist_input, in_dist_input.reshape((in_dist_input.shape[0], -1))), axis=1)
            
            if regressor_feature == 'ODIN': # because we do not need to iterate on multiple layers for ODIN
                break
            
    for out_dist in config['model_params']['out_dist_list']:
        if out_dist == 'subset_cifar100':
            out_test_loader = data_loader.getNonTargetDataSet(out_dist, config['trainer_params']['batch_size'], in_transform, config['exp_params']['dataroot'], idx=config['model_params']['out_idx'], num_oods=config['model_params']['num_oods'])
        # elif out_dist == 'stl10':
        #     tf_stl10 = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(config['model_params']['transform_params']['mean'],config['model_params']['transform_params']['std'])])
        #     out_test_loader = data_loader.getNonTargetDataSet(out_dist, config['trainer_params']['batch_size'], tf_stl10, config['exp_params']['dataroot'], idx=config['model_params']['out_idx'], num_oods=config['model_params']['num_oods'])
        else:
            out_test_loader = data_loader.getNonTargetDataSet(out_dist, config['trainer_params']['batch_size'], in_transform, config['exp_params']['dataroot'])
        #print('Out-distribution: ' + out_dist)
        
        init_reg_out = True
        for regressor_feature in regressor_features:
            if  regressor_feature == 'knn_mahalanobis_class_cov' or regressor_feature == 'knn_mahalanobis_tied_cov':
                continue
            # num_output is the number of layers
            for i in range(num_output):
                out_dist_input = get_features_for_regressor(regressor_feature, model, config, out_test_loader, out_dist,  i, False, device,
                                                    class_mean, class_precision, tied_mean, tied_precision, pca_list, knn_search_list=[None]*num_output, knn_mean=None, knn_precision=None,
                                                    knn_conformance_search_list=knn_conformance_search_list, conformance_labels=conformance_labels, k=0)

                #print("out_dist_input shape- ", out_dist_input.shape)
                out_dist_input = np.asarray(out_dist_input, dtype=np.float32)
                #print("Mean score at layer {} for regression type {}: {}".format(i,regressor_feature,np.mean(out_dist_input)))
                if init_reg_out:
                    regressor_out_dist_input = out_dist_input.reshape((out_dist_input.shape[0], -1))
                    init_reg_out = False
                else:
                    regressor_out_dist_input = np.concatenate((regressor_out_dist_input, out_dist_input.reshape((out_dist_input.shape[0], -1))), axis=1)
                

                if regressor_feature == 'ODIN': # because we do not need to iterate on multiple layers for ODIN
                    break

        if knn_feature_flag:
            for k in num_knns:
                # In case we only have Knn features
                if only_knn_feature:
                    init_reg_in = True
                    init_reg_out = True

                class_mean, class_precision, tied_precision, knn_search_list, knn_mean, knn_precision = knn_inputs[k]
                # calculate for in_dist
                additional_cols = 0
                for i in range(num_output):
                    in_dist_input = get_features_for_regressor(knn_feature_name, model, config, test_loader, config['exp_params']['dataset'], i, True, device,
                                                        class_mean, class_precision, tied_mean, tied_precision, pca_list, knn_search_list, knn_mean, knn_precision,
                                                        knn_conformance_search_list, conformance_labels, k)

                    #print("in_dist_input shape- ", in_dist_input.shape)
                    in_dist_input = np.asarray(in_dist_input, dtype=np.float32)

                    #print("Mean score at layer {} for regression type {}: {}".format(i,regressor_feature,np.mean(in_dist_input)))

                    if init_reg_in:
                        regressor_in_dist_input = in_dist_input.reshape((in_dist_input.shape[0], -1))
                        init_reg_in = False
                    else:
                        regressor_in_dist_input = np.concatenate((regressor_in_dist_input, in_dist_input.reshape((in_dist_input.shape[0], -1))), axis=1)

                    additional_cols+=1

                # calculate for out_dist
                for i in range(num_output):
                    out_dist_input = get_features_for_regressor(knn_feature_name, model, config, out_test_loader, out_dist,  i, False, device,
                                                        class_mean, class_precision, tied_mean, tied_precision, pca_list, knn_search_list, knn_mean, knn_precision,
                                                        knn_conformance_search_list, conformance_labels, k)

                    #print("out_dist_input shape- ", out_dist_input.shape)
                    out_dist_input = np.asarray(out_dist_input, dtype=np.float32)
                    #print("Mean score at layer {} for regression type {}: {}".format(i,regressor_feature,np.mean(out_dist_input)))
                    if init_reg_out:
                        regressor_out_dist_input = out_dist_input.reshape((out_dist_input.shape[0], -1))
                        init_reg_out = False
                    else:
                        regressor_out_dist_input = np.concatenate((regressor_out_dist_input, out_dist_input.reshape((out_dist_input.shape[0], -1))), axis=1)


                regressor_in_dist_input = np.asarray(regressor_in_dist_input, dtype=np.float32)
                regressor_out_dist_input = np.asarray(regressor_out_dist_input, dtype=np.float32)
                Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(regressor_out_dist_input, regressor_in_dist_input)
                file_name = os.path.join(config['logging_params']['outf'], 'Mahalanobis_%s_%s_%s_%s.npy' % (str(m_list[0]), str(k), config['exp_params']['dataset'] , out_dist))
                Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
                np.save(file_name, Mahalanobis_data)
                regressor_in_dist_input = regressor_in_dist_input[:,:-additional_cols]
                regressor_out_dist_input = regressor_out_dist_input[:,:-additional_cols]

        else:
            regressor_in_dist_input = np.asarray(regressor_in_dist_input, dtype=np.float32)
            regressor_out_dist_input = np.asarray(regressor_out_dist_input, dtype=np.float32)
            Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(regressor_out_dist_input, regressor_in_dist_input)
            file_name = os.path.join(config['logging_params']['outf'], 'Mahalanobis_%s_%s_%s_%s.npy' % (str(m_list[0]), str(0), config['exp_params']['dataset'] , out_dist))
            Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
            np.save(file_name, Mahalanobis_data)


def plot_tsne(config, device, model, in_data_loader, out_data_loader, ood_labels, layer_index): 

    utils.dump_features_for_given_layer(train_loader=in_data_loader, 
                                  model=model, 
                                  layer_index=layer_index, 
                                  out_file=config['tsne_params']['feature_file_list'][0], # assuming the first item in the 'feature_file_list' is the in-dist file name and the second is the OOD feature's file name
                                  device=device)

    utils.dump_features_for_given_layer(train_loader=out_data_loader, 
                                  model=model, 
                                  layer_index=layer_index, 
                                  out_file=config['tsne_params']['feature_file_list'][1], # assuming the first item in the 'feature_file_list' is the in-dist file name and the second is the OOD feature's file name
                                  device=device)

    utils.plot_TSNE(n_components=config['tsne_params']['n_components'], verbose=config['tsne_params']['verbose'], perplexity=config['tsne_params']['perplexity'], 
                    n_iter=config['tsne_params']['n_iter'], random_state = config['tsne_params']['random_state'], color_list=config['tsne_params']['color_list'], class_list=config['tsne_params']['class_list'], 
                   feature_file_list=config['tsne_params']['feature_file_list'], ood_labels = ood_labels, num_train_ood=config['tsne_params']['num_train_ood'],
                   plot_file_name=config['tsne_params']['plot_file_name']) 

if __name__ == '__main__':
    # setting all gpus visible to this program
    os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3"

    # parse command line arguments for the config file
    config = parse_args()

    # if gpu is available then set device=gpu else set device=cpu
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format((config['trainer_params']['gpu'])))
        torch.cuda.manual_seed(0)
    else:
        device = torch.device('cpu')

    # set the path to pre-trained model and output
    pre_trained_net = config['model_params']['pretrained_model_path']
    if os.path.isdir(config['logging_params']['outf']) == False:
        os.mkdir(config['logging_params']['outf'])

    # create model
    if config['model_params']['name']!='ResNet50':
        model = classifier_models[config['model_params']['name']](config['exp_params']['num_classes'])
    else:
        model = classifier_models[config['model_params']['name']]()

    # load pretrained model
    model.load_state_dict(torch.load(pre_trained_net, map_location = device))

    model.to(device)
    print('load model: ' + config['model_params']['net_type'])

    # load dataset
    #print('load target data: ', config['exp_params']['dataset'])
    #print("Mean: ", type(config['model_params']['transform_params']['mean']), config['model_params']['transform_params']['mean'])
    #print("Std: ", type(config['model_params']['transform_params']['std']), config['model_params']['transform_params']['std'])

    if config['model_params']['net_type']=='lenet5':
        #print("in_transform for: ", config['model_params']['net_type'])
        IMG_SIZE = config['model_params']['img_size']
        in_transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                 transforms.ToTensor()])
    else:
        in_transform = transforms.Compose([transforms.Resize(config['model_params']['img_size']), transforms.ToTensor(), transforms.Normalize(config['model_params']['transform_params']['mean'],config['model_params']['transform_params']['std'])])

    train_loader, test_loader = data_loader.getTargetDataSet(config['exp_params']['dataset'], config['trainer_params']['batch_size'], in_transform, config['exp_params']['dataroot'])

    main(config, device, model, in_transform, train_loader, test_loader)


    # for plotting t-SNE plot with SCIFAR100
    # out_test_loader = data_loader.getNonTargetDataSet(config['model_params']['out_dist_list'][0], config['trainer_params']['batch_size'], in_transform, config['exp_params']['dataroot'], idx=config['model_params']['out_idx'], num_oods=config['model_params']['num_oods'])

    # ood_label_file_name = config['tsne_params']['ood_labels_file']
    # ood_labels = np.load(ood_label_file_name)['indices']
    # plot_tsne(config, device, model, train_loader, out_test_loader, ood_labels, layer_index=4)
