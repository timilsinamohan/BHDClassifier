__author__ = 'mohan'
import pandas as pd
import networkx as nx
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold
import time
from scipy import sparse
from gbssl import LGC,HMN,PARW,CAMLP,OMNIProp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import lil_matrix
from rescal import rescal_als
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from nodevectors import Node2Vec
import stellargraph as sg
from stellargraph import StellarGraph
import tensorflow as tf
from tensorflow.keras import Model
from stellargraph.mapper import RelationalFullBatchNodeGenerator
from stellargraph.layer import RGCN
from numpy.random import seed
import networkx as nx
from nodevectors import Node2Vec
import pickle
from functools import reduce
from stellargraph.mapper import KGTripleGenerator
from stellargraph.layer import ComplEx
from tensorflow.keras import callbacks, optimizers, losses, metrics, regularizers, Model
from tensorly.decomposition import non_negative_parafac,tucker
#seed(1)

tf.random.set_seed(2)
np.random.seed(20)


def get_ground_truth(data):
    
    with open('Datasets/'+data+'/labels.pkl','rb') as f:
        labels = pickle.load(f)
        train_nodes_target = {}
        test_nodes_target = {}
        vaild_nodes_target = {}
        all_node_labels = {}
        cnt = 0
        for u,v in labels[0]:
            train_nodes_target[u]=v
            all_node_labels[u] = v
            cnt+=1
            
        for u,v in labels[2]:
            test_nodes_target[u]=v
            all_node_labels[u] = v
            cnt+=1
            
        for u,v in labels[1]:
            vaild_nodes_target[u]=v
            all_node_labels[u] = v
            cnt+=1
            
        
    return train_nodes_target,test_nodes_target,vaild_nodes_target,all_node_labels



def get_camlp_score(graph_matx,train_nodes,labs):

    camlp = CAMLP(graph=graph_matx)
    camlp.fit(np.array(train_nodes),np.array(labs))

    P =  camlp.predict_proba(np.arange(n))
    labels = camlp.predict(np.arange(n))
    return P.T,labels


def get_lgc_score(graph_matx,train_nodes,labs):

    #alp = get_best_alpha(Y,train_idx)
    alp = 0.1
    lgc = LGC(graph=graph_matx,alpha=alp)

    lgc.fit(np.array(train_nodes),np.array(labs))

    P =  lgc.predict_proba(np.arange(n))
    labels = lgc.predict(np.arange(n))
    return P.T,labels

def get_harmonic_score(train_nodes,labs,graph_matx):

    hmn = HMN(graph=graph_matx)
    hmn.fit(np.array(train_nodes),np.array(labs))

    P =  hmn.predict_proba(np.arange(n))
    labels = hmn.predict(np.arange(n))
   
    return P.T,labels

def get_omni_score(graph_matx, train_nodes,labs):

    omni = OMNIProp(graph=graph_matx)
    omni.fit(np.array(train_nodes),np.array(labs))
    P =  omni.predict_proba(np.arange(n))
    labels = omni.predict(np.arange(n))

    omni.fit(np.array(train_nodes),np.array(labs))

    return P.T,labels

def get_parw_score(train_nodes,labs):

    parw = PARW(graph=graph_matx,lamb=10)
    parw.fit(np.array(train_nodes),np.array(labs))

    P =  parw.predict_proba(np.arange(n))
    labels = parw.predict(np.arange(n))
    return P.T,labels



def get_heat_score(graph_matx,train_nodes,labs):

    n_samples = graph_matx.shape[0]
    n_classes = l
    B = np.zeros((n_samples,n_classes))
    B[train_nodes,labs] = 1
    L = sparse.csgraph.laplacian(graph_matx,normed=True)
    I = np.eye(n,n,dtype=np.float64)
    t = 1.0
    iteration = 30
    V = I - (t/iteration) * L
    state_matrix = B.copy()

    for j in range(iteration):
        state_matrix = V.dot(state_matrix)

    state_matrix = np.array(state_matrix)

    labels = np.argmax(state_matrix,axis=1)

    return state_matrix.T,labels

def get_best_time(train_nodes,graph_matx,valid_nodes,valid_labels):
    n_samples = graph_matx.shape[0]
    
    time = [0.001,0.01,1.0,10.0,100.0]
    best_results = {}
    n_classes = l
    for t in time:
        B = np.zeros((n_samples,n_classes))
        B[train_nodes,train_labels] = 1
        B[valid_nodes,valid_labels] = np.nan
        col_mean = np.nanmean(B, axis=0)
        inds = np.where(np.isnan(B))
        B[inds] = np.take(col_mean, inds[1])
        L = sparse.csgraph.laplacian(graph_matx,normed=True)
        I = np.eye(n,n,dtype=np.float64)
        iteration = 30
        V = I - (t/iteration) * L
        state_matrix_from_harmonic,labs = get_harmonic_score(train_nodes,train_labels,graph_matx)
        
        C = B - state_matrix_from_harmonic.T
        state_matrix = C.copy()
        for j in range(iteration):
            state_matrix = V.dot(state_matrix)
            state_matrix = state_matrix + state_matrix_from_harmonic.T
        state_matrix = np.array(state_matrix)
        labels = np.argmax(state_matrix,axis=1)
        acc = accuracy_score(valid_nodes, valid_labels)
        best_results[t] = acc
    kL = sorted(best_results, key= best_results.get, reverse=True)
    print ("Best Time:", kL[0])
    return kL[0]
        
        

def get_bhd_score(graph_matx,train_nodes,labs):

    n_samples = graph_matx.shape[0]
    n_classes = l
    B = np.zeros((n_samples,n_classes))
    B[train_nodes,labs] = 1
    B[test_nodes,test_labels] = np.nan
    col_mean = np.nanmean(B, axis=0)
    inds = np.where(np.isnan(B))
    B[inds] = np.take(col_mean, inds[1])

    L = sparse.csgraph.laplacian(graph_matx,normed=True)
    I = np.eye(n,n,dtype=np.float64)
    t = get_best_time(train_nodes,graph_matx,validation_labels,validation_labels)
    iteration = 30
    V = I - (t/iteration) * L
    #state_matrix = B.copy()
    state_matrix_from_harmonic,labs = get_harmonic_score(train_nodes,labs,graph_matx)
    C = B - state_matrix_from_harmonic.T

    state_matrix = C.copy()


    for j in range(iteration):
        state_matrix = V.dot(state_matrix)
        state_matrix = state_matrix + state_matrix_from_harmonic.T


    state_matrix = np.array(state_matrix)

    labels = np.argmax(state_matrix,axis=1)

    return state_matrix.T,labels


def match(result,test,n):
    correct = 0
    incorrect = 0
    c = 0
    i = 0
    while i < len(result) and c < n:
        nid = result[i][0]
        label = result[i][1][0]
        conf = result[i][1][1]
        if not nid in test:
            i += 1
            continue
        test_label = test[nid]
        if label == test_label:
            correct += 1
        else:
            incorrect += 1
        c += 1
        i += 1
    p = correct / float(correct + incorrect)
    r = (correct+incorrect) / float(len(test))
    return p,r


def get_knn_graph(labels,EMB,train_nodes,test_nodes):
    k_range = range(1,12)
    param_grid = dict(n_neighbors = k_range)
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn,param_grid, cv = 10, scoring = "accuracy")
    grid.fit(EMB[train_nodes],labels)
    GF = kneighbors_graph(EMB,grid.best_params_['n_neighbors'], mode='connectivity',include_self=False)
    return GF


def get_rgcn_embeddings(data):
         # load data
        
    with open('Datasets/'+data+'/edges.pkl','rb') as f:
        edgelist = pickle.load(f)
    Layers = len(edgelist)
    source = []
    target = []
    rels =  []
    for i in range(Layers):
        r,c = edgelist[i].nonzero()
        source.append(list(r))
        target.append(list(c))
        rels.append([i]*len(r))
            
    source = reduce(lambda x,y: x+y, source)
    target = reduce(lambda x,y: x+y,target)
    rels = reduce(lambda x,y: x+y,rels)
    df = pd.DataFrame({"source":source , "target": target,"LayerID":rels})
      
    ixp = list(set(list(set(df["source"].values)) + list(set(df["target"].values))))
    I = np.identity(len(ixp))
    features = pd.DataFrame(data = I, index = ixp)
    graph = sg.StellarGraph(
    edges= df, edge_type_column="LayerID", nodes=features)

    rgcn_generator = RelationalFullBatchNodeGenerator(graph, sparse=True)
    rgcn_model = RGCN(layer_sizes=[150], activations=["relu"], generator=rgcn_generator)
    x_in, x_out = rgcn_model.in_out_tensors()
    embedding_model = Model(inputs=x_in, outputs=x_out)
    all_embeddings = embedding_model.predict(rgcn_generator.flow(graph.nodes()))
    X = all_embeddings.squeeze(0)
   
    return X
    

def get_node2vec_embeddings(data):
        # load data    
    with open('Datasets/'+data+'/edges.pkl','rb') as f:
        edgelist = pickle.load(f)
    Layers = len(edgelist)
    source = []
    target = []
    rels =  []
    for i in range(Layers):
        r,c = edgelist[i].nonzero()
        source.append(list(r))
        target.append(list(c))
        rels.append([i]*len(r))
            
    source = reduce(lambda x,y: x+y, source)
    target = reduce(lambda x,y: x+y,target)
    rels = reduce(lambda x,y: x+y,rels)
    df = pd.DataFrame({"source":source , "target": target,"LayerID":rels})
      
    
    G= nx.from_pandas_edgelist(df, 'source', 'target', ["LayerID"])
    emb = 150
    g2v = Node2Vec(n_components=emb)
    g2v.fit(G)
    nodes = list(G.nodes)
    X = np.zeros((len(nodes),emb))
    cnt = 0
    for i in nodes:
        X[cnt,:] = g2v.predict(i)
        cnt+=1 

    return X


def get_tucker_embeddings(data):

        # load data
    with open('Datasets/'+data+'/edges.pkl','rb') as f:
        edges = pickle.load(f)
           
    np.random.seed(1)
    m,m = edges[0].shape
    T = np.zeros((m,m,len(edges)))
    for i in range(len(edges)):
        T[edges[i].nonzero()[0],edges[i].nonzero()[1],i]=1
    R = 150
    
    core, emb = tucker(T, ranks=[R, 4, 2])
    
    return emb[0]



def get_tensor_embeddings(data):

        # load data
    with open('Datasets/'+data+'/edges.pkl','rb') as f:
        edges = pickle.load(f)
           
    np.random.seed(1)
    A, R, _, _, _ = rescal_als(edges, 150, init='nvecs', conv=1e-3,lambda_A=10.0, lambda_R=10.0)
    A = np.abs(A)
    
    return A


def innerfold(train_nodes,test_nodes):
    true_labels = np.array(GT[:])
    train_nodes = np.array(train_nodes)
    test_nodes = np.array(test_nodes)
    print ("graph construction started")
    graph = get_knn_graph(train_labels,graph_embeddings,train_nodes,test_nodes)
    print ("graph constructed")
    #score,label_pred = get_harmonic_score(train_nodes,train_labels,graph)
    #score,label_pred = get_lgc_score(graph,train_nodes,train_labels)
    #score,label_pred = get_camlp_score(graph,train_nodes,train_labels)
    #score,label_pred = get_omni_score(graph,train_nodes,train_labels)
    #score,label_pred = get_heat_score(graph,train_nodes,train_labels)
    score,label_pred = get_bhd_score(graph,train_nodes,train_labels)

    results = {}
    test = {}
    cnt = 0
    for nodes in test_nodes:
        results[nodes]=(label_pred[cnt],score[label_pred[cnt],nodes])
        test[nodes] = true_labels[cnt]
        cnt+=1
    
    sorted_results = sorted(results.items(), key=lambda x:x[1][1], reverse=True)
    prec_at_ten,rec_at_k = match(sorted_results, test, int(len(test_nodes)*0.1))
    prec_at_all,rec_at_all = match(sorted_results, test, int(len(test_nodes)*1.0))
    acc = accuracy_score(test_labels, label_pred[test_nodes])
    prec_at_ten = np.round_(prec_at_ten,decimals= 2)
    prec_at_all = np.round_(prec_at_all,decimals= 2)
    F_score = 2*rec_at_all* prec_at_all/( prec_at_all + rec_at_all)
    acc = np.round_(acc,decimals= 2)
   
    return prec_at_ten,prec_at_all,acc,F_score


if __name__ == '__main__':
    data_src = "ACM"
    True_Train_Labels,True_Test_Labels,True_Valid_Labels, all_nodes_label = get_ground_truth(data_src)
    graph_embeddings = get_tensor_embeddings(data_src)
    #graph_embeddings = get_node2vec_embeddings(data_src)
    #graph_embeddings = get_rgcn_embeddings(data_src)
    #graph_embeddings = get_tucker_embeddings(data_src)
    print ("graph_embedding is calculated")
    n = graph_embeddings.shape[0]
    train_nodes = list(True_Train_Labels.keys())
    train_labels = list(True_Train_Labels.values())
    test_nodes = list(True_Test_Labels.keys())
    test_labels = list(True_Test_Labels.values())
    validation_nodes = list(True_Valid_Labels.keys())
    validation_labels = list(True_Valid_Labels.values())
    all_nodes_label = dict(sorted(all_nodes_label.items()))
    l = len(list(set(train_labels)))
    GT = list(all_nodes_label.values())
    prec_ten,prec_all, acc_test,F_score = innerfold(train_nodes, test_nodes)
    print ("Prec at 10:",prec_ten)
    print ("Prec at 100:",prec_all)
    print ("Accuracy:",acc_test)
    print ("F-Score:",F_score)

