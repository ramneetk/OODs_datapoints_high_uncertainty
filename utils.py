from __future__ import print_function

import os
import numpy as np
import torch
import pickle

from PIL import Image
from matplotlib import cm


from datasketch import MinHashLSHForest
from datasketch import MinHash

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from scipy.spatial.distance import pdist, cdist, squareform

from sklearn.decomposition import PCA
import sklearn.covariance

import argparse
import data_loader
import calculate_log as callog
import models
import lib_generation

from torchvision import transforms
from torch.autograd import Variable

from annoy import AnnoyIndex

from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans

#from pynndescent import NNDescent

import random
random.seed(20)


class KNNSearch:
    def __init__(self,features,kwargs):

        self.org_features = features
        if kwargs["normalize"]:
            self.features  = preprocessing.normalize(features,norm='l2')
        else:
            self.features  = features

        self.kwargs    = kwargs
        self.predictor = None
    
    def fit(self):
        if self.kwargs['algorithm'] =='datasketch':
            self.__datasketch_fit()
        elif self.kwargs['algorithm']=='annoy':
            self.__annoy_fit()
        elif self.kwargs['algorithm']=='exact':
            self.__exhaustive_fit()
        elif self.kwargs['algorithm']=='falconn':
            self.__falconn_fit()
        # elif self.kwargs['algorithm']=='descent':
        #     self.__descent_fit()
        elif self.kwargs['algorithm']=='random':
            self.__random_fit()
        else:
            raise Exception("Algorithm=[{}] not yet implemented".format(self.kwargs['algorithm']))

    def predict(self,input,k):
        if self.kwargs['algorithm'] =='datasketch':
            return self.__datasketch_predict(input,k)
        elif self.kwargs['algorithm']=='annoy':
            return self.__annoy_predict(input,k)
        elif self.kwargs['algorithm']=='exact':
            return self.__exhaustive_predict(input,k)
        elif self.kwargs['algorithm']=='falconn':
            return self.__falconn_predict(input,k)
        # elif self.kwargs['algorithm']=='descent':
        #     return self.__descent_predict(input,k)
        elif self.kwargs['algorithm']=='random':
            return self.__random_predict(input,k)
        else:
            raise Exception("Algorithm=[{}] not yet implemented".format(self.kwargs['algorithm']))

    def __datasketch_fit(self):
        if self.kwargs['create']:
            # Create a list of MinHash objects
            min_hash_obj_list = []
            forest = MinHashLSHForest(num_perm=self.kwargs['num_perm'])
            for i in range(len(self.features)):
                min_hash_obj_list.append(MinHash(num_perm=self.kwargs['num_perm']))
                for d in self.features[i]:
                    min_hash_obj_list[i].update(d)
                forest.add(i, min_hash_obj_list[i])
            # IMPORTANT: must call index() otherwise the keys won't be searchable
            forest.index()   
            with open(self.kwargs['file_path'],"wb") as f:
                pickle.dump(forest, f)
                pickle.dump(min_hash_obj_list, f)
            self.predictor = [forest,min_hash_obj_list]
        else:
            with open(self.kwargs['file_path'], "rb") as f:
                forest = pickle.load(f)
                min_hash_obj_list = pickle.load(f)
                self.predictor = [forest,min_hash_obj_list]

    def __datasketch_predict(self,input,k):
        forest,min_hash_obj_list = self.predictor
        if type(input)==int:
            return forest.query(min_hash_obj_list[input], k)
        else:
            min_hash_obj = MinHash(num_perm=self.kwargs['num_perm'])
            for d in input:
                min_hash_obj.update(d)
            return forest.query(min_hash_obj, k)

    def __annoy_fit(self):
        if self.kwargs['create']:
            indexer = AnnoyIndex(self.features.shape[1],self.kwargs['metric'])
            for i,f in enumerate(self.features):
                indexer.add_item(i,f)
            indexer.build(self.kwargs['num_trees'])
            indexer.save(self.kwargs['file_path'])
            self.predictor = indexer
        else:
            forest = AnnoyIndex(self.features.shape[1], self.kwargs['metric'])
            forest.load(self.kwargs['file_path'])
            self.predictor = forest
    
    def __annoy_predict(self,input,k):
        annoy_forest = self.predictor
        if type(input)==int:
            return annoy_forest.get_nns_by_item(input, k, search_k=-1, include_distances=False)
        else:
            return annoy_forest.get_nns_by_vector(input, k, search_k=-1, include_distances=False)

    def __exhaustive_fit(self):
        self.predictor = NearestNeighbors(algorithm='ball_tree')
        self.predictor.fit(self.features)
    
    def __exhaustive_predict(self,input,k):
        if type(input)==int:
            return self.predictor.kneighbors(self.features[input].reshape(1,-1),n_neighbors=k,return_distance=False)[0]
        else:
            return self.predictor.kneighbors(input.reshape(1,-1),n_neighbors=k,return_distance=False)[0]
    
    def __falconn_fit(self):
        """
        Initializes locality-sensitive hashing with FALCONN to find nearest neighbors in training data.
        """

        import falconn

        dimension = self.features.shape[1]
        nb_tables = self.kwargs['nb_tables']
        number_bits = self.kwargs['number_bits']

        # LSH parameters
        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = dimension
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp.l = nb_tables
        params_cp.num_rotations = 2  # for dense set it to 1; for sparse data set it to 2
        params_cp.seed = 5721840
        # we want to use all the available threads to set up
        params_cp.num_setup_threads = 0
        params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable

        # we build number_bits-bit hashes so that each table has
        # 2^number_bits bins; a rule of thumb is to have the number
        # of bins be the same order of magnitude as the number of data points
        falconn.compute_number_of_hash_functions(number_bits, params_cp)
        self._falconn_table = falconn.LSHIndex(params_cp)
        self._falconn_query_object = None
        self._FALCONN_NB_TABLES = nb_tables

        # Center the dataset and the queries: this improves the performance of LSH quite a bit.
        self.center    = np.mean(self.features, axis=0)
        self.features -= self.center

        # add features to falconn table
        self._falconn_table.setup(self.features)

    def __falconn_predict(self,input,k):

        # Normalize input if you care about the cosine similarity
        if type(input)==int:
            input = self.features[input]
        else:
            if self.kwargs['normalize']:
                input /= np.linalg.norm(input)
                # Center the input and the queries: this improves the performance of LSH quite a bit.
                input -= self.center        

        # Late falconn query_object construction
        # Since I suppose there might be an error
        # if table.setup() will be called after
        if self._falconn_query_object is None:
            self._falconn_query_object = self._falconn_table.construct_query_object()
            self._falconn_query_object.set_num_probes(
                self._FALCONN_NB_TABLES
            )

        query_res = self._falconn_query_object.find_k_nearest_neighbors(input,k)
        return query_res

    # def __descent_fit(self):
    #     self.predictor = NNDescent(data=self.features, metric=self.kwargs['metric'])
    
    # def __descent_predict(self,input,k):
    #     input = np.expand_dims(input, axis=0) # input should be an array of search points
    #     index = self.predictor
    #     return index.query(input, k)[0][0] # returns indices of NN, distances of the NN from the input

    def __random_fit(self):
        pass
    
    def __random_predict(self,input,k):
        rand_index_list = []
        for i in range(k):
            rand_index_list.append(random.randint(0,len(self.features)-1))

        return rand_index_list

def get_precision(features):
    '''
    Compute precision
    '''
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    #group_lasso = sklearn.covariance.MinCovDet(assume_centered=False,support_fraction=1)
    group_lasso.fit(features)
    precision = group_lasso.precision_
    return precision

def dump_subsetcifar_100_images():
    '''
    function for dumping dataset's features got from the layer with index 'layer_index' 
    in the model the features will be dumped in the output file out_file.
    '''
    indices = [6, 14, 68, 71]

    in_transform = transforms.Compose([transforms.ToTensor()])

    out_test_loader = data_loader.getNonTargetDataSet('subset_cifar100', 100, in_transform, './data', idx=indices, num_oods=2000)

    images = []

    counter = 0

    for data, labels in out_test_loader:
        for img in data:
            img = np.transpose(img.numpy(), (1, 2, 0))
            im = Image.fromarray(np.uint8(img*255))
            im.save('subset-cifar100-images/{}.png'.format(counter))
            counter += 1
            

if __name__ == "__main__":
    dump_subsetcifar_100_images()

def dump_features_for_given_layer(train_loader, 
                                  model, 
                                  layer_index, 
                                  out_file, 
                                  device):
    '''
    function for dumping dataset's features got from the layer with index 'layer_index' 
    in the model the features will be dumped in the output file out_file.
    '''
    feature_list = [] 
    label_list   = []
        
    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)

        # fetch features of the data from the intermediate layer of the model
        features = model.intermediate_forward(data, layer_index)
        features = features.view(features.size(0), features.size(1), -1)
        features = torch.mean(features.data, 2)
        labels   = labels.detach().cpu().numpy()
        features = features.detach().cpu().numpy()
        label_list.append(labels)
        feature_list.append(features)
        
    #print("features len: ", len(feature_list))
    all_labels = np.concatenate(label_list).flatten()
    all_features = np.concatenate(feature_list)
    np.savez(out_file, labels = all_labels, features = all_features) 

def get_all_features(model, 
                    num_classes, 
                    feature_list, 
                    train_loader, 
                    device):
    model.eval()
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    
    labels = []
    for data, target in train_loader:
        labels.append(target.cpu().numpy())
        # total += data.size(0)
        data = data.to(device)
        output, out_features = model.feature_list(data)
        
        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            
        # compute the accuracy
        # pred = output.data.max(1)[1]
        # equal_flag = pred.eq(target.to(device)).cpu()
        # correct += equal_flag.sum()
        
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
        
    # print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))
    return list_features,np.concatenate(labels)

def plot_TSNE_results(tsne_results,
                     labels):

    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:brown', 'tab:cyan', 'tab:orange', 'tab:pink', 'k']

    plt.figure(figsize=(5,4))

    fig, ax = plt.subplots()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    colours = ListedColormap(color_list)
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=labels, cmap=colours,alpha=.25)
    # plt.legend(handles=scatter.legend_elements()[0], labels=class_list,loc='lower left',
    #        fontsize=5)
    plt.title("T-SNE")
    # plt.savefig('{}.pdf'.format(plot_file_name))

def plot_TSNE(n_components, 
              verbose, 
              perplexity, 
              n_iter, 
              random_state,
              color_list, 
              class_list, 
              feature_file_list,
              ood_labels, 
              num_train_ood,
              plot_file_name):
    '''
    function to plot the TSNE map to visualize the features in low-dimensional space
    '''
    print("random_state: ", random_state)
    for i,f in enumerate(feature_file_list):
        input_features_file_data = np.load(f)
        if i == 0:
            print("i- ", i)
            features = input_features_file_data['features']
            labels = input_features_file_data['labels']
        else:
            print("i- ", i)
            all_ood_features = input_features_file_data['features'][num_train_ood:]
            desired_oods = all_ood_features[ood_labels]
            features = np.vstack([features,desired_oods])
            # ood_labels = np.ndarray(len(input_features_file_data['features']))
            ood_labels[:] = 10

            print("***ood_labels shape ", ood_labels.shape)
            print("Unique ood labels ", np.unique(ood_labels))
            labels = np.concatenate([labels, ood_labels]).flatten()
            print("After concat, labels shape ",labels.shape)
            print("Unique total labels ", np.unique(labels))

    # ----------------- T-SNE -------------------
    tsne = TSNE(n_components=n_components, verbose=verbose, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    tsne_results = tsne.fit_transform(features)
    print ("tsne_results shape: ", tsne_results.shape)
    np.savez('tsne_results.npz',results=tsne_results,labels=labels)

    plt.figure(figsize=(5,4))

    fig, ax = plt.subplots()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    colours = ListedColormap(color_list)
    scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=labels, cmap=colours,alpha=.25)
    plt.legend(handles=scatter.legend_elements()[0], labels=class_list,loc='lower left',
           fontsize=5)
    plt.title("T-SNE")
    plt.savefig('{}.pdf'.format(plot_file_name))

    # --------- T-SNE on K-means clustering --------
    # kmeans = KMeans(n_clusters=len(class_list), random_state=0, max_iter=1000).fit(features)
    # plt.figure(figsize=(5,4))
    # scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=kmeans.labels_, cmap=colours)
    # plt.legend(handles=scatter.legend_elements()[0], labels=class_list)
    # plt.title("T-SNE based on K-means clustering")
    # plt.savefig('T-SNE based on K-means clustering')

def get_mean_tied_precision(list_features, feature_list, num_classes, device):
    sample_class_mean = []
    out_count = 0
    num_output = len(feature_list)
    #print("feature list in get_mean_prec: ", feature_list)
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).to(device)
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1
        
    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
                
        # find inverse 
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False) 
        #group_lasso = sklearn.covariance.MinCovDet(assume_centered=False,support_fraction=1)       
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().to(device)
        #print(temp_precision.shape)
        precision.append(temp_precision)
        
    return sample_class_mean, precision

def get_tied_mean_tied_precision(list_features, feature_list, num_classes, device):
    sample_class_mean = []
    out_count = 0
    num_output = len(feature_list)
    for num_feature in feature_list:
        layer_mean = torch.mean(torch.cat(list_features[out_count], dim=0), 0)
        temp_list = torch.Tensor(num_classes, int(num_feature)).to(device)
        for j in range(num_classes):
            temp_list[j] = layer_mean
        sample_class_mean.append(temp_list)
        out_count += 1
        
    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
                
        # find inverse 
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False) 
        #group_lasso = sklearn.covariance.MinCovDet(assume_centered=False,support_fraction=1)       
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().to(device)
        #print(temp_precision.shape)
        precision.append(temp_precision)
        
    #print("Mean: ", sample_class_mean)
    #print("Precesion: ", precision)
    return sample_class_mean, precision

def calc_tied_mean_tied_precision(model, 
                                   num_classes, 
                                   feature_list, 
                                   train_loader, 
                                   device):
    '''
    function to compute class wise sample mean and tied precision (inverse of covariance)
    Authors: Mahalanobis Paper
    '''

    list_features, _ = get_all_features(model, num_classes, feature_list, train_loader, device)
    sample_mean, precision = get_tied_mean_tied_precision(list_features, feature_list, num_classes, device)        
            
    return sample_mean, precision

def calc_class_mean_tied_precision(model, 
                                   num_classes, 
                                   feature_list, 
                                   train_loader, 
                                   device):
    '''
    function to compute class wise sample mean and tied precision (inverse of covariance)
    Authors: Mahalanobis Paper
    '''

    list_features, _ = get_all_features(model, num_classes, feature_list, train_loader, device)
    sample_class_mean, precision = get_mean_tied_precision(list_features, feature_list, num_classes, device)        
            
    return sample_class_mean, precision

def get_class_mean_class_precision(list_features, feature_list, num_classes, device):
    sample_class_mean = []
    out_count = 0
    num_output = len(feature_list)
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).to(device)
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    # class_wise_precision is the list for storing class wise precision for different layers
    # it will be a list of list- for each layer, precision for each class
    class_wise_precision = []
    for k in range(num_output):
        class_wise_X = []
        for i in range(num_classes):
            class_wise_X.append(list_features[k][i] - sample_class_mean[k][i])
                
        # calculate class wise precision for the layer k
        class_wise_precision_for_layer_k = []
        for i in range(num_classes):
            group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
            #group_lasso = sklearn.covariance.MinCovDet(assume_centered=False,support_fraction=1)
            group_lasso.fit(class_wise_X[i].cpu().numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float().to(device)
            class_wise_precision_for_layer_k.append(temp_precision)
        class_wise_precision.append(class_wise_precision_for_layer_k)
    
    return sample_class_mean, class_wise_precision

def calc_class_mean_class_precision(model, 
                                    num_classes, 
                                    feature_list, 
                                    train_loader, 
                                    device):
    '''
    function to  compute class wise sample mean and class wise precision (inverse of covariance)
    Modified from original code calc_class_mean_tied_precision
    '''

    
    list_features,_ = get_all_features(model, num_classes, feature_list, train_loader, device)
    sample_class_mean, class_wise_precision = get_class_mean_class_precision(list_features, feature_list, num_classes, device)
    
    return sample_class_mean, class_wise_precision

def modify_features_using_knn(features, 
                              knn_search,  
                              knn_args,
                              k):
    '''
    Find nearest neighbors for input samples and create new features according to kwargs
    '''

    #k = knn_args['k']
    modified_features = [[] for _ in range(len(features))]
    
    if type(features) == torch.Tensor:
        device = features.device
    else:
        device = None

    # iterate over features of this class
    for i,feat in enumerate(features):
        if device:
            neighbors = knn_search.predict(feat.detach().cpu().numpy(), k)
        else:
            neighbors = knn_search.predict(feat, k)

        # keep original feature in modified feature
        if knn_args['keep_original']:
            modified_features[i].append(feat)

        # keep mean of nearest neighbors in modified feature
        if knn_args['keep_knn_mean']:         
            knn_mean = np.mean(knn_search.org_features[neighbors], 0)
            if device:
                knn_mean = torch.from_numpy(knn_search).to(device)
            modified_features[i].append(knn_mean)
        
        #keep std of nearest neighbors in modified feature
        if knn_args['keep_knn_std']:
            # group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
            # group_lasso.fit(knn_search.org_features[neighbors])
            # knn_covariance = group_lasso.covariance_
            # knn_covariance_diagonal = knn_covariance.diagonal()
            # knn_std_diagonal = np.sqrt(knn_covariance_diagonal).astype(np.float32)

            knn_features = knn_search.org_features[neighbors]
            if device:
                feat = feat.detach().cpu().numpy().reshape(1,-1)
            
            #knn_with_original_features = np.vstack((feat,knn_features))
            knn_std_diagonal = np.sqrt(np.sum(np.square(knn_features-feat),axis=0))
            #knn_std_diagonal = np.std(knn_with_original_features, axis = 0)

            if device:
                knn_std_diagonal = torch.from_numpy(knn_std_diagonal).to(device)
            
            modified_features[i].append(knn_std_diagonal)

        if device:
            modified_features[i] = torch.cat(modified_features[i]).reshape(1,-1)
        else:
            modified_features[i] = np.concatenate(modified_features[i]).reshape(1,-1)

    if device:
        modified_features = torch.cat(modified_features,dim=0)
    else:
        modified_features = np.concatenate(modified_features,axis=0)
    
    return modified_features

def calc_knn_mean_precision(model,
                            num_classes,
                            feature_list,
                            train_loader,
                            device,
                            cov_type,
                            knn_type_args,
                            knn_args,
                            k):

    all_knn_search = []

    list_features, _ = get_all_features(model, num_classes, feature_list, train_loader, device)
    list_features_modified = [[] for _ in range(len(list_features))]
    # iterating on features in each layer
    for l_idx,features in enumerate(list_features):

        for idx,features_class in enumerate(features):
            features[idx] = features_class.detach().cpu().numpy()

        # Create object for K-NN search algorithm
        knn_search = KNNSearch(np.concatenate(features,axis=0),knn_type_args)
        knn_search.fit()
        all_knn_search.append(knn_search)

        # modified features for each class 
        for features_class in features:     
            list_features_modified[l_idx].append(torch.from_numpy(modify_features_using_knn(features_class, knn_search, knn_args,k)).to(device))

    if cov_type =='tied_cov':
        sample_class_mean, precision = get_mean_tied_precision(list_features_modified, feature_list, num_classes, device) 
    elif cov_type =='class_cov':
        sample_class_mean, precision = get_class_mean_class_precision(list_features_modified, feature_list, num_classes, device)


    return all_knn_search, sample_class_mean, precision


def calc_knnSearch_and_labels(model,
                            num_classes,
                            feature_list,
                            train_loader,
                            device,
                            knn_type_args):

    all_knn_search = []

    list_features, labels = get_all_features(model, num_classes, feature_list, train_loader, device)
    # iterating on features in each layer
    for features in list_features:

        for idx,features_class in enumerate(features):
            features[idx] = features_class.detach().cpu().numpy()

        # Create object for K-NN search algorithm
        knn_search = KNNSearch(np.concatenate(features,axis=0),knn_type_args)
        knn_search.fit()
        all_knn_search.append(knn_search)

    return all_knn_search, labels
