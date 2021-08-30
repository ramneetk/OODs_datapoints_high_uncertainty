from __future__ import print_function
import torch
import numpy as np
import scipy.stats
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from scipy.spatial.distance import pdist, cdist, squareform

from sklearn.decomposition import PCA
from datasketch import MinHashLSHForest
from datasketch import MinHash

import sklearn.covariance
import utils

# lid of a batch of query points X
def mle_batch(data, batch, k):
    '''
    commpute lid score using data & batch with k-neighbors
    return: a: computed LID score
    '''
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

# this function is from https://github.com/xingjunm/lid_adversarial_subspace_detection
def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y

def get_pca(model, num_classes, feature_list, train_loader, device):
    """
    return: pca_list: list of class-wise precision objects
    """    
    model.eval()

    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    
    for data, target in train_loader:
        data = data.to(device)
        output, out_features = model.feature_list(data)
        
        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
        
        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1                
            num_sample_per_class[label] += 1

    # pca_list[k][i] with contain PCA object for kth layer and ith class
    pca_list = []
    for k in range(num_output):
        class_wise_pca = []
        for i in range(num_classes):
            pca = PCA(n_components=None)
            pca.fit(list_features[k][i].detach().cpu().numpy())
            class_wise_pca.append(pca)
        pca_list.append(class_wise_pca)
    
    return pca_list

def get_pca_scores(features, pca_list, layer_index, class_num, k):
 
    transformed_features = pca_list[layer_index][class_num].transform(features.detach().cpu().numpy())   
    # get the L2 norm of the last k values of the transformed features for the class
    transformed_feature_last_k_values = transformed_features[:,transformed_features.shape[1]-k:]
    #print("Transformed for class ", i, "- ", transformed_class_last_k_features[0])
    pca_scores = - np.linalg.norm(transformed_feature_last_k_values, axis=1)
    pca_scores_tensor = torch.from_numpy(pca_scores)  

    return pca_scores_tensor

def get_gradient_using_mahalanobis_score(data, out_features, config, cov_type, sample_mean, precision, layer_index, device):
    num_classes = config['exp_params']['num_classes']
    
    # compute Mahalanobis score
    gaussian_score = 0
    for i in range(num_classes):
        batch_sample_mean = sample_mean[layer_index][i]
        zero_f = out_features.data - batch_sample_mean

        if cov_type == 'tied_cov':
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        else:
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index][i]), zero_f.t()).diag()   

        if i == 0:
            gaussian_score = term_gau.view(-1,1)
        else:
            gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
    
    # Input_processing
    pure_gau = 0
    sample_pred = gaussian_score.max(1)[1]
    batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
    zero_f = out_features - Variable(batch_sample_mean)


    if cov_type == 'tied_cov':
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()  
    else:
        for i in range(len(sample_pred)):
            if i == 0: 
                pure_gau = -0.5*torch.mm(torch.mm(zero_f[i].unsqueeze(0), Variable(precision[layer_index][sample_pred[i].item()])), zero_f[i].unsqueeze(0).t()).diag()
            else:
                pure_gau = torch.cat((pure_gau, -0.5*torch.mm(torch.mm(zero_f[i].unsqueeze(0), Variable(precision[layer_index][sample_pred[i].item()])), zero_f[i].unsqueeze(0).t()).diag()), 0)

    loss = torch.mean(-pure_gau)
    loss.backward()
        
    gradient =  torch.ge(data.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / config['exp_params']['noise_params']['gradient_params'][0])
    gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / config['exp_params']['noise_params']['gradient_params'][1])
    
    if config['exp_params']['dataset'] != 'toy_data' :
        gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / config['exp_params']['noise_params']['gradient_params'][2])

    return gradient

def get_pca_score(model, config, test_loader, out_flag, 
                  sample_mean, precision, layer_index, magnitude, pca_list, device):
    '''
    Compute the PCA score on input dataset
    return: PCA score from layer_index
    '''
    model.eval()
    PCA = []

    # parse config file for inputs
    num_classes = config['exp_params']['num_classes']
    outf = config['logging_params']['outf'] 
    net_type = config['model_params']['net_type']
    cov_type = 'class_cov' # for PCA it makes sense to use class-cov

    if out_flag == True:
        temp_file_name = '%s/confidence_Ga%s_In.txt'%(outf, str(layer_index))
    else:
        temp_file_name = '%s/confidence_Ga%s_Out.txt'%(outf, str(layer_index))
        
    g = open(temp_file_name, 'w')

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        # HARD CODE
        out_features = model.intermediate_forward(data, config['model_params']['penul_layer'])
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        # add noise to the input features according to Mahalanobis method- therefore we require class-wise mean and class-wise precision 
        gradient = get_gradient_using_mahalanobis_score(data, out_features, config, cov_type, sample_mean, precision, layer_index, device) 
        noisy_inputs = torch.add(data.data, -magnitude, gradient)

        # HARD CODE
        noise_out_features = model.intermediate_forward(noisy_inputs, config['model_params']['penul_layer'])
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)

        # compute PCA scores
        noise_gaussian_score = 0
        for i in range(num_classes):
            term_gau = get_pca_scores(noise_out_features.data, pca_list, layer_index, i, int(0.6*(noise_out_features.data.shape[1])))  
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        PCA.extend(noise_gaussian_score.cpu().numpy())
        for i in range(data.size(0)):
            g.write("{}\n".format(noise_gaussian_score[i]))
    g.close()

    return PCA


def get_Mahalanobis_score(reg_feature, model, config, test_loader, out_flag, mean_for_noise, 
                          precision_for_noise, sample_mean, precision, layer_index, 
                          magnitude, knn_search, device, knn, k):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []
    
    # parse config file for inputs
    num_classes = config['exp_params']['num_classes']
    outf = config['logging_params']['outf'] 
    net_type = config['model_params']['net_type']
    if reg_feature == 'mahalanobis_tied_cov' or reg_feature == 'knn_mahalanobis_tied_cov' or reg_feature == 'mahalanobis_tied_mean_tied_cov':
        cov_type = 'tied_cov'
        #print("***** ", reg_feature, cov_type)
    elif reg_feature == 'mahalanobis_class_cov' or reg_feature == 'knn_mahalanobis_class_cov':
        cov_type = 'class_cov'
        #print("***** ", reg_feature, cov_type)
    else:
        raise Exception("Wrong type for regressor_feature with Mahalanobis score. Valid options are: {}, {}".format("mahalanobis_tied_cov","mahalanobis_class_cov"))


    if out_flag == True:
        temp_file_name = '%s/confidence_Ga%s_In.txt'%(outf, str(layer_index))
    else:
        temp_file_name = '%s/confidence_Ga%s_Out.txt'%(outf, str(layer_index))
        
    g = open(temp_file_name, 'w')

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        # HARD CODE
        out_features = model.intermediate_forward(data, config['model_params']['penul_layer'])
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        # add noise to the input features
        gradient = get_gradient_using_mahalanobis_score(data, out_features, config, cov_type, mean_for_noise, precision_for_noise, layer_index, device) 
        noisy_inputs = torch.add(data.data, -magnitude, gradient)

        # HARD CODE
        noise_out_features = model.intermediate_forward(noisy_inputs, config['model_params']['penul_layer'])
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)

        if knn:
            knn_args = config['exp_params']['knn_args']
            noise_out_features = utils.modify_features_using_knn(noise_out_features,
                                                           knn_search,
                                                           knn_args,
                                                           k)
            # if out_flag == True:
            #     std_nns_file_name = '%s/std_NNs%s_In.txt'%(outf, str(layer_index))
            # else:
            #     std_nns_file_name = '%s/std_NNs%s_Out.txt'%(outf, str(layer_index))
                
            # std_nns_file = open(std_nns_file_name, 'w')

            # noise_out_features_numpy = noise_out_features
            # noise_out_features_numpy = noise_out_features_numpy.detach().cpu().numpy()
            # for i in range(noise_out_features_numpy.size(0)):
            #     std_nns_file.write("{}\n".format(noise_out_features_numpy[i]))
            # std_nns_file.close()

        # compute Mahalanobis scores
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            if cov_type == 'tied_cov': #original scoring mechanism
                term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            elif cov_type == 'class_cov':
                term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index][i]), zero_f.t()).diag()
            else:
                raise Exception("Wrong cov_type for computing Mahalanobis score. Valid options are: {},{}".format("tied_cov","class_cov"))
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())
        for i in range(data.size(0)):
            g.write("{}\n".format(noise_gaussian_score[i]))
    g.close()

    return Mahalanobis

def get_knn_conformance_score(model, config, test_loader, out_flag, layer_index,
                              knn_conformance_search, conformance_labels, device, k):
    
    model.eval()
    scores = []
    
    #print("*******k=", k)
    # parse config file for inputs
    num_classes = config['exp_params']['num_classes']
    outf = config['logging_params']['outf'] 
    net_type = config['model_params']['net_type']

    if out_flag == True:
        temp_file_name = '%s/confidence_Ga%s_In.txt'%(outf, str(layer_index))
    else:
        temp_file_name = '%s/confidence_Ga%s_Out.txt'%(outf, str(layer_index))
        
    g = open(temp_file_name, 'w')

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        # HARD CODE
        out_features = model.intermediate_forward(data, config['model_params']['penul_layer'])
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        for feat in out_features:
            neighbors = knn_conformance_search.predict(feat.detach().cpu().numpy(),k)
            labels_neighbors  = conformance_labels[neighbors]
            label_frequencies = np.unique(labels_neighbors,return_counts=True)[1]
            scores.append(scipy.stats.entropy(label_frequencies))
            g.write("{}\n".format(scores[-1]))

    g.close()

    return scores
    

def get_posterior(model, net_type, test_loader, magnitude, temperature, outf, out_flag, device):

    '''
    Compute the maximum value of (processed) posterior distribution - ODIN
    return: null
    '''
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    if out_flag == True:
        temp_file_name_val = '%s/confidence_PoV_In.txt'%(outf)
        temp_file_name_test = '%s/confidence_PoT_In.txt'%(outf)
    else:
        temp_file_name_val = '%s/confidence_PoV_Out.txt'%(outf)
        temp_file_name_test = '%s/confidence_PoT_Out.txt'%(outf)
        
    g = open(temp_file_name_val, 'w')
    f = open(temp_file_name_test, 'w')
    
    softmax_outputs = np.ndarray((0,1))

    for data, _ in test_loader:
        total += data.size(0)
        data = data.to(device)
        data = Variable(data, requires_grad = True)
        batch_output = model(data)
            
        # temperature scaling
        outputs = batch_output / temperature
        labels = outputs.data.max(1)[1]
        labels = Variable(labels)
        loss = criterion(outputs, labels)
        loss.backward()
         
        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if net_type == 'densenet':
            gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (63.0/255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (62.1/255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (66.7/255.0))
        elif net_type == 'resnet':
            gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (0.2010))

        tempInputs = torch.add(data.data,  -magnitude, gradient)
        outputs = model(Variable(tempInputs, volatile=True))
        outputs = outputs / temperature
        soft_out = F.softmax(outputs, dim=1)
        soft_out, _ = torch.max(soft_out.data, dim=1)
        
        
        for i in range(data.size(0)):
            if total <= 1000:
                g.write("{}\n".format(soft_out[i]))
            else:
                f.write("{}\n".format(soft_out[i]))

        soft_out = soft_out.view(-1,1)
        
        soft_out = soft_out.detach().cpu().numpy()
        softmax_outputs = np.vstack((softmax_outputs,soft_out))
                
    f.close()
    g.close()

    #print("softmax_outputs_shape: ", softmax_outputs.shape)
    return softmax_outputs

    
def get_Mahalanobis_score_adv(model, test_data, test_label, num_classes, outf, net_type, sample_mean, precision, layer_index, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on adversarial samples
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []
    batch_size = 100
    total = 0
    
    for data_index in range(int(np.floor(test_data.size(0)/batch_size))):
        target = test_label[total : total + batch_size].to(device)
        data = test_data[total : total + batch_size].to(device)
        total += batch_size
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
         
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if net_type == 'densenet':
            gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (63.0/255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (62.1/255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (66.7/255.0))
        elif net_type == 'resnet':
            gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (0.2010))
        tempInputs = torch.add(data.data, -magnitude, gradient)
 
        noise_out_features = model.intermediate_forward(Variable(tempInputs, volatile=True), layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())
        
    return Mahalanobis


def get_LID(model, test_clean_data, test_adv_data, test_noisy_data, test_label, num_output):
    '''
    Compute LID score on adversarial samples
    return: LID score
    '''
    model.eval()  
    total = 0
    batch_size = 100
    
    LID, LID_adv, LID_noisy = [], [], []    
    # overlap_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    overlap_list = [50]
    for i in overlap_list:
        LID.append([])
        LID_adv.append([])
        LID_noisy.append([])

    #print("size of input data",test_clean_data.size())    
    for data_index in range(int(np.floor((test_clean_data.size(0)+batch_size-1)/batch_size))):
        data = test_clean_data[total : total + batch_size].to(device)
        adv_data = test_adv_data[total : total + batch_size].to(device)
        noisy_data = test_noisy_data[total : total + batch_size].to(device)
        target = test_label[total : total + batch_size].to(device)

        total += data.size(0)
        data, target = Variable(data, volatile=True), Variable(target)
        
        output, out_features = model.feature_list(data)
        X_act = []
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2).detach().cpu()
            X_act.append(np.asarray(out_features[i], dtype=np.float32).reshape((out_features[i].size(0), -1)))
        
        output, out_features = model.feature_list(Variable(adv_data, volatile=True))
        X_act_adv = []
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2).detach().cpu()
            X_act_adv.append(np.asarray(out_features[i], dtype=np.float32).reshape((out_features[i].size(0), -1)))

        output, out_features = model.feature_list(Variable(noisy_data, volatile=True))
        X_act_noisy = []
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2).detach().cpu()
            X_act_noisy.append(np.asarray(out_features[i], dtype=np.float32).reshape((out_features[i].size(0), -1)))
        
        # LID
        list_counter = 0 
        for overlap in overlap_list:
            LID_list = []
            LID_adv_list = []
            LID_noisy_list = []

            for j in range(num_output):
                lid_score = mle_batch(X_act[j], X_act[j], k = overlap)
                lid_score = lid_score.reshape((lid_score.shape[0], -1))
                lid_adv_score = mle_batch(X_act[j], X_act_adv[j], k = overlap)
                lid_adv_score = lid_adv_score.reshape((lid_adv_score.shape[0], -1))
                lid_noisy_score = mle_batch(X_act[j], X_act_noisy[j], k = overlap)
                lid_noisy_score = lid_noisy_score.reshape((lid_noisy_score.shape[0], -1))
                
                LID_list.append(lid_score)
                LID_adv_list.append(lid_adv_score)
                LID_noisy_list.append(lid_noisy_score)

            LID_concat = LID_list[0]
            LID_adv_concat = LID_adv_list[0]
            LID_noisy_concat = LID_noisy_list[0]

            for i in range(1, num_output):
                LID_concat = np.concatenate((LID_concat, LID_list[i]), axis=1)
                LID_adv_concat = np.concatenate((LID_adv_concat, LID_adv_list[i]), axis=1)
                LID_noisy_concat = np.concatenate((LID_noisy_concat, LID_noisy_list[i]), axis=1)
            
            LID[list_counter].extend(LID_concat)
            LID_adv[list_counter].extend(LID_adv_concat)
            LID_noisy[list_counter].extend(LID_noisy_concat)
            list_counter += 1
            
    return LID, LID_adv, LID_noisy
