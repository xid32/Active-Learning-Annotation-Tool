# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:02:54 2021

@author: farza
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from time import time
from sklearn import preprocessing
import xgboost as xgb




# Load dataset

path = os.getcwd()
files = os.listdir(path)
files_xls = [f for f in files if f[-4:] == 'xlsx']
# testname = files_xls.pop(0)
# testset = pd.read_excel(testname)

trainset = pd.DataFrame()
for f in files_xls:
    data = pd.read_excel(f)
    trainset = trainset.append(data)
    
#        #####################################################################################################################################
    
# standardizing the training and test set
   
trainset = trainset.reset_index(drop=True)
train_generalized = trainset.drop(['date_exp','start','end'], axis = 1)

y_train = train_generalized.pop('label')



std_scale = preprocessing.StandardScaler().fit(train_generalized)
X_train = std_scale.transform(train_generalized)
X_train = pd.DataFrame(X_train)

# test_generalized = testset.drop(['date_exp','start','end'], axis = 1)
# y_test = test_generalized.pop('label')
# X_test = std_scale.transform(test_generalized)
# X_test = pd.DataFrame(X_test)

#print the shape of the X_train
print('Shape Train:\t{}'.format(X_train.shape))
#print('Shape Test:\t{}\n'.format(X_test.shape))


# preparing X_train for t-SNE transformation
tsne_data = X_train
label = y_train   
label_counts = label.value_counts()


t0_clust = time()
# Reduce dimensions (speed up) and perform PCA + t-SNE prior to K-means clustering
pca = PCA(n_components=0.9, random_state=3)
tsne_data = pca.fit_transform(tsne_data)
    
# Transform data
tsne = TSNE(n_components = 3, random_state=3)
tsne_transformed = tsne.fit_transform(tsne_data)
    
    
    
    
# K-means clustering to for accounting for sample diversity   
kmeans = KMeans(init="random", n_clusters=6, n_init=10, max_iter=300, random_state=42)
kmeans.fit(tsne_transformed)
kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }
clust_label = pd.DataFrame(kmeans.labels_, columns = ["clus_label"])
clustered_train_data = pd.concat([X_train, clust_label], axis=1)


# How much time is spent in t-SNE + clustering
t1_clust = time()
t_clust = t1_clust - t0_clust

# selecting the right K for clustering -- for the right query budget
# A list holds the silhouette coefficients for each k
#silhouette_coefficients = []
    
# Notice you start at 2 clusters for silhouette coefficient
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#     kmeans.fit(tsne_transformed)
#     score = silhouette_score(tsne_transformed, kmeans.labels_)
#     silhouette_coefficients.append(score)
# plt.style.use("fivethirtyeight")
# plt.plot(range(2, 11), silhouette_coefficients)
# plt.xticks(range(2, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette Coefficient")
# plt.show()




# Xgbosst Parameters in classification in each iteration
settings = {}
xgb_param = {}
xgb_param["seed"] = 123
xgb_param["objective"] = "multi:softprob"
xgb_param["lambda"] = 0.8
xgb_param["eta"] = 0.05
xgb_param["max_depth"] = 1
xgb_param["subsample"] = 0.8
xgb_param["min_child_weight"] = 5
xgb_param["silent"] = 0
xgb_param["num_class"] = 2   
xgb_param["nthread"] = 32

settings["xgb_param"] = xgb_param
settings["xgb_training_num_boost_round"] = 1000   
settings["xgb_training_early_stopping_rounds"] = 200  
settings["xgb_training_maximize"] = False   
settings["xgb_training_verbose_eval"] = False  


# calculating classification performance with F1-score -- creating confusion matrix
def eval_model(model_prob,x_pred,y_truth):

    #what we care more about is the positive fscore. We don't care about the null
    y_pred = np.argmax(model_prob, axis=1)
    f1 = f1_score(y_truth, y_pred, average='binary')
    MCC = matthews_corrcoef(y_truth, y_pred)
    confusion_mat = confusion_matrix(y_truth, y_pred)
    return {'f1':f1, 'MCC': MCC, 'y_pred': y_pred,'y_truth': y_truth, 'confusion_mat':confusion_mat }

#----------------------------------------------------------------------------------------------------------------------------------------
    
# preparing the data for Clustered Entropy Active LEarning method
# max_entquery_list[6:,:] : it is the list of the samples nominated to be labelled                    
X_ent_clust = X_train
clustered_tr_data = clustered_train_data

# Start time for the Clustered entropy method  
t0_clust_maxent = time()

# Clustered Entropy methodd for Active Learning

#init_query_list = X_train.sample(n=5, replace = False)
init_query_list = pd.DataFrame()
for i in range (0,6):
    clust_subset = clustered_tr_data[clustered_tr_data.clus_label == i]
    sample = clust_subset.sample(n=1, replace = False)
    init_query_list = pd.concat([init_query_list, sample], axis=0)
redun = init_query_list.pop('clus_label')

clustered_tr_data = clustered_tr_data.drop(init_query_list.index)
X_ent_clust = X_ent_clust.drop(init_query_list.index)

d_matrix_train = xgb.DMatrix(np.asmatrix(init_query_list), label=y_train[init_query_list.index])
watchlist = [(d_matrix_train, "train")]

max_entropy_model = xgb.train(settings["xgb_param"], d_matrix_train, evals=watchlist, num_boost_round=settings["xgb_training_num_boost_round"],
early_stopping_rounds=settings["xgb_training_early_stopping_rounds"], maximize=settings["xgb_training_maximize"],
verbose_eval=settings["xgb_training_verbose_eval"],
)
d_test = xgb.DMatrix(np.asmatrix(X_ent_clust), label=y_train[X_ent_clust.index])
predictions = max_entropy_model.predict(d_test)



unlabeled_predictions = np.sum(predictions * np.log(predictions + 1e-10), axis=1)
unlabeled_predictions = pd.DataFrame(unlabeled_predictions)
unlabeled_predictions.index = X_ent_clust.index
#unlabeled_predictions = pd.concat([unlabeled_predictions, sample], axis=0)


#selected_indices = unlabeled_predictions.nsmallest(5,0).index

kmeans_label = clustered_tr_data[['clus_label']]
clus_labeled_predictions = pd.concat([unlabeled_predictions,kmeans_label], axis = 1)
selected_indices = pd.DataFrame()
for i in range (0,6):
    cluslab_subset = clus_labeled_predictions[clus_labeled_predictions.clus_label == i] 
    indices = cluslab_subset.nsmallest(1,0).index
    indices = pd.DataFrame(indices)
    selected_indices = pd.concat([selected_indices, indices], axis = 0)


#selected_indices = np.argpartition(unlabeled_predictions, 5)[:5]
iteration_time = {}
batch_eval_ent = {}
max_entquery_list = X_ent_clust.loc[selected_indices.loc[:,0]]
X_ent_clust = X_ent_clust.drop(max_entquery_list.index)
clustered_tr_data = clustered_tr_data.drop(max_entquery_list.index)
for i in range (1,12):
    
#    t0_iteration_maxent = time()
    
    d_matrix_train = xgb.DMatrix(np.asmatrix(max_entquery_list), label=y_train[max_entquery_list.index])
    watchlist = [(d_matrix_train, "train")]

    max_entropy_model = xgb.train(settings["xgb_param"], d_matrix_train, evals=watchlist, num_boost_round=settings["xgb_training_num_boost_round"],
                                  early_stopping_rounds=settings["xgb_training_early_stopping_rounds"], maximize=settings["xgb_training_maximize"],
                                  verbose_eval=settings["xgb_training_verbose_eval"],
                                  )
    d_test = xgb.DMatrix(np.asmatrix(X_ent_clust), label=y_train[X_ent_clust.index])
    predictions = max_entropy_model.predict(d_test)        
  
    unlabeled_predictions = np.sum(predictions * np.log(predictions + 1e-10), axis=1)
    unlabeled_predictions = pd.DataFrame(unlabeled_predictions)
    unlabeled_predictions.index = X_ent_clust.index
    
    kmeans_label = clustered_tr_data[['clus_label']]
    clus_labeled_predictions = pd.concat([unlabeled_predictions,kmeans_label], axis = 1)
    selected_indices = pd.DataFrame()
    for j in range (0,6):
        cluslab_subset = clus_labeled_predictions[clus_labeled_predictions.clus_label == j] 
        indices = cluslab_subset.nsmallest(1,0).index
        indices = pd.DataFrame(indices)
        selected_indices = pd.concat([selected_indices, indices], axis = 0)
#    selected_indices = np.argpartition(unlabeled_predictions, 5)[:5]




    max_entquery_list = pd.concat([max_entquery_list, X_ent_clust.loc[selected_indices.loc[:,0]]], axis=0)
    X_ent_clust = X_ent_clust.drop(selected_indices.loc[:,0])    
    clustered_tr_data = clustered_tr_data.drop(selected_indices.loc[:,0])
    
#    d_matrix_test = xgb.DMatrix(np.asmatrix(X_test), label=y_test)
#    max_entropy_model_prob = max_entropy_model.predict(d_matrix_test)
#    score_entropy = eval_model(max_entropy_model_prob,X_test,y_test)
#    batch_eval_ent[i] = score_entropy
#    t1_iteration_maxent = time()
#    t_iteration_maxent = (t1_iteration_maxent - t0_iteration_maxent)
#    iteration_time[i] = {'time':t_iteration_maxent}
t1_clust_maxent = time()
#total_iteration_time = pd.concat([total_iteration_time, pd.DataFrame([eval['time'] for eval in list(iteration_time.values())])], axis = 1)
#Clust_max_ent_avg = pd.concat([Clust_max_ent_avg, pd.DataFrame([eval['f1'] for eval in list(batch_eval_ent.values())])], axis = 1)
#t_clust_maxent = pd.concat([t_clust_maxent, pd.DataFrame([t1_clust_maxent - t0_clust_maxent])], axis = 0)

#        #####################################################################################################################################
    


print(selected_indices)

import code
code.InteractiveConsole(locals=globals()).interact()

