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
from tensorly.decomposition import non_negative_parafac,tucker
seed(1)

tf.random.set_seed(2)



np.random.seed(20)


def get_ground_truth():

    df = pd.read_csv("Datasets/CKM-Physicians-Innovation_nodes.txt",
                      sep = " ")
    src = df["nodeClubs"].values
    dest = df["nodeID"].values
    
    df_graph = pd.read_csv("Datasets/CKM-Physicians-Innovation_multiplex.edges",
                        sep = " ",
                       header = None,
                       names=['LayerID', 'nodeID_1','nodeID_2',
                              'weight'])
    
    Nodes = np.unique(np.concatenate([pd.unique(df_graph['nodeID_1']),pd.unique(df_graph['nodeID_2'])]))
    common_nodes = np.intersect1d(dest,Nodes)
    labels = []
    for index, row in df.iterrows():
        if (row["nodeID"] in common_nodes):
            labels.append(row["nodeClubs"])
    
    labels = np.array(labels)
    ##there is 9 term in src changing this to 2 means we have 3 labels now 0,1,2
    labels[np.where(labels==9)]=2
    
    return labels



def get_camlp_score(graph_matx,train_nodes,labs):

    camlp = CAMLP(graph=graph_matx)
    camlp.fit(np.array(train_nodes),np.array(labs))

    P =  camlp.predict_proba(np.arange(n))
    labels = camlp.predict(np.arange(n))
    return P.T,labels


def get_lgc_score(graph_matx,train_nodes,labs):

    #alp = get_best_alpha(Y,train_idx)
    alp = 0.99
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

def get_best_time(train_nodes,graph_matx):
    GTL  = np.array(True_Labels[:])
    n_samples = graph_matx.shape[0]
    #time = np.arange(0.05,5.05,0.05)
    time = np.array([0.00001,0.0001,0.001,0.01,1.0,10.0,100.0])
    cv_results = {}
    old_train_nodes = train_nodes
    train_labels = GTL[train_nodes]
    for t in time:
        FOLDS = 2
        acc = np.zeros(FOLDS)
        skf = StratifiedKFold(n_splits=FOLDS)
        cnt = 0
        for test_nodes, train_nodes in skf.split(old_train_nodes, train_labels):
            n_classes = l
            B = np.zeros((n_samples,n_classes))
            B[train_nodes,train_labels[train_nodes]] = 1
            B[test_nodes,True_Labels[test_nodes]] = np.nan
            col_mean = np.nanmean(B, axis=0)
            inds = np.where(np.isnan(B))
            B[inds] = np.take(col_mean, inds[1])

            L = sparse.csgraph.laplacian(graph_matx,normed=True)
            I = np.eye(n,n,dtype=np.float64)

            iteration = 30
            V = I - (t/iteration) * L
            state_matrix_from_harmonic,labs = get_harmonic_score(train_nodes,train_labels[train_nodes],graph_matx)
            C = B - state_matrix_from_harmonic.T
            state_matrix = C.copy()

            for j in range(iteration):
                state_matrix = V.dot(state_matrix)
                state_matrix = state_matrix + state_matrix_from_harmonic.T

            state_matrix = np.array(state_matrix)
            labels = np.argmax(state_matrix,axis=1)

            acc[cnt] = accuracy_score(GTL[test_nodes], labels[test_nodes])
            cnt+=1
        cv_results[t] = acc.mean()

    kL = sorted(cv_results, key=cv_results.get, reverse=True)
    print ("Best Time:", kL[0])
    return kL[0]

def get_bhd_score(graph_matx,train_nodes,labs):


    n_samples = graph_matx.shape[0]
    n_classes = l
    B = np.zeros((n_samples,n_classes))
    B[train_nodes,labs] = 1
    test_nodes = np.setdiff1d(np.arange(n_samples),train_nodes)
    B[test_nodes,True_Labels[test_nodes]] = np.nan
    col_mean = np.nanmean(B, axis=0)
    inds = np.where(np.isnan(B))
    B[inds] = np.take(col_mean, inds[1])

    L = sparse.csgraph.laplacian(graph_matx,normed=True)
    I = np.eye(n,n,dtype=np.float64)
    t = get_best_time(train_nodes,graph_matx)
    #t = 2.0
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
    EMB_train = EMB[train_nodes]
    k_range = range(1,12)
    param_grid = dict(n_neighbors = k_range)
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn,param_grid, cv = 10, scoring = "accuracy")
    grid.fit(EMB_train,labels)
    GF = kneighbors_graph(EMB,grid.best_params_['n_neighbors'], mode='connectivity',include_self=False)
    return GF



def get_baseline():
        # load data
        # load data
    df = pd.read_csv("Datasets/CKM-Physicians-Innovation_multiplex.edges",
                     sep = " ",
                       header = None,
                       names=['LayerID', 'nodeID_1','nodeID_2',
                              'weight'])
    G= nx.from_pandas_edgelist(df, 'nodeID_1', 'nodeID_2', ["LayerID"],create_using=nx.MultiGraph())
    X = nx.to_scipy_sparse_matrix(G, nodelist = G.nodes())
    
    return X

def get_rgcn_embeddings():
         # load data
        
    df = pd.read_csv("Datasets/CKM-Physicians-Innovation_multiplex.edges",
                     sep = " ",
                       header = None,
                       names=['LayerID', "source","target",
                              'weight'])
    
   
    #df = pd.DataFrame({"source":source , "target": target,"orientation":orientation})
    #N = len(list(set(df["source"].values)) + list(set(df["target"].values)))
    ixp = list(set(list(set(df["source"].values)) + list(set(df["target"].values))))
    I = np.identity(len(ixp))
    features = pd.DataFrame(data = I, index = ixp)
    graph = sg.StellarGraph(
    edges= df, edge_type_column="LayerID", nodes=features
        )

    rgcn_generator = RelationalFullBatchNodeGenerator(graph, sparse=True)
    rgcn_model = RGCN(layer_sizes=[150], activations=["relu"], generator=rgcn_generator)
    x_in, x_out = rgcn_model.in_out_tensors()
    embedding_model = Model(inputs=x_in, outputs=x_out)
    all_embeddings = embedding_model.predict(rgcn_generator.flow(graph.nodes()))
    X = all_embeddings.squeeze(0)
   
    return X
    

def get_node2vec_embeddings():
        # load data
        # load data
    df = pd.read_csv("Datasets/CKM-Physicians-Innovation_multiplex.edges",
                     sep = " ",
                       header = None,
                       names=['LayerID', 'nodeID_1','nodeID_2',
                              'weight'])
    G= nx.from_pandas_edgelist(df, 'nodeID_1', 'nodeID_2', ["LayerID"])
    emb = 128
    g2v = Node2Vec(n_components=emb)
    g2v.fit(G)
    nodes = list(G.nodes)
    X = np.zeros((len(nodes),emb))
    cnt = 0
    for i in nodes:
        X[cnt,:] = g2v.predict(i)
        cnt+=1   
    return X

def get_tucker_embeddings():

        # load data
    df = pd.read_csv("Datasets/CKM-Physicians-Innovation_multiplex.edges",
                     sep = " ",
                       header = None,
                       names=['LayerID', 'nodeID_1','nodeID_2',
                              'weight'])
    
    Layers = pd.unique(df['LayerID'])
    Nodes = np.unique(np.concatenate([pd.unique(df['nodeID_1']),pd.unique(df['nodeID_2'])]))
    IDX = list(Nodes)
    L = len(Layers)
    m = len(Nodes)
    X = [sparse.lil_matrix((m,m)) for i in range(L)]
           
    np.random.seed(1)
    T = np.zeros((m,m,len(X)))
    for i in range(len(X)):
        T[X[i].nonzero()[0],X[i].nonzero()[1],i]=1
    R = 150

    core, emb = tucker(T, ranks=[R, 4, 2])
   
    return emb[0]

def get_tensor_embeddings():

        # load data
    df = pd.read_csv("Datasets/CKM-Physicians-Innovation_multiplex.edges",
                     sep = " ",
                       header = None,
                       names=['LayerID', 'nodeID_1','nodeID_2',
                              'weight'])
    
    Layers = pd.unique(df['LayerID'])
    Nodes = np.unique(np.concatenate([pd.unique(df['nodeID_1']),pd.unique(df['nodeID_2'])]))
    IDX = list(Nodes)
    L = len(Layers)
    m = len(Nodes)
    X = [sparse.lil_matrix((m,m)) for i in range(L)]
   
    for index, row in df.iterrows():
        
        X[row['LayerID']-1][IDX.index(row['nodeID_1']),IDX.index(row['nodeID_2'])] = 1
       
   
    A, R, _, _, _ = rescal_als(X, 150, init='nvecs', conv=1e-3,lambda_A=10, lambda_R=10)
    
    A = np.abs(A)
    return A

def innerfold(train_nodes,test_nodes):
    true_labels = np.array(True_Labels[:])
    train_nodes = np.array(train_nodes)
    test_nodes = np.array(test_nodes)
    labels = true_labels[train_nodes]
    mask_labels = true_labels.copy()
    mask_labels[test_nodes] = 0
    train_labels = true_labels[train_nodes]
    graph = get_knn_graph(train_labels,graph_embeddings,train_nodes,test_nodes)
    #graph = get_baseline()

    #score,label_pred = get_harmonic_score(train_nodes,labels,graph)
    #score,label_pred = get_lgc_score(graph,train_nodes,labels)
    #score,label_pred = get_camlp_score(graph,train_nodes,labels)
    #score,label_pred = get_omni_score(graph,train_nodes,labels)
    #score,label_pred = get_heat_score(graph,train_nodes,labels)
    score,label_pred = get_bhd_score(graph,train_nodes,labels)

    results = {}
    test = {}
    cnt = 0
    for nodes in test_nodes:
            results[nodes]=(label_pred[cnt],score[label_pred[cnt],nodes])
            test[nodes] = true_labels[nodes]
            cnt+=1
   
    sorted_results = sorted(results.items(), key=lambda x:x[1][1], reverse=True)
    prec_at_10,rec_at_k = match(sorted_results, test, int(len(test_nodes)*0.1))
    prec_at_100,rec_at_k = match(sorted_results, test, int(len(test_nodes)*1.0))
    acc = accuracy_score(true_labels[test_nodes], label_pred[test_nodes])
 

    return prec_at_10,prec_at_100,acc



if __name__ == '__main__':
    True_Labels = get_ground_truth()
    graph_embeddings = get_tensor_embeddings()
    #graph_embeddings = get_node2vec_embeddings()
    #graph_embeddings = get_rgcn_embeddings()
    #graph_embeddings = get_tucker_embeddings()
    n = graph_embeddings.shape[0]
    l = len(list(set(True_Labels)))
    nodes = range(len(True_Labels))
    FOLDS = 10
    Accuracy = np.zeros(FOLDS)
    # Do 10 random trails ###
    acc_test = np.zeros(FOLDS)
    prec_test_10 = np.zeros(FOLDS)
    prec_test_100 = np.zeros(FOLDS)
    skf = StratifiedKFold(n_splits=FOLDS)
    cnt = 0
    for test_nodes, train_nodes in skf.split(nodes, True_Labels):
        prec_test_10[cnt],prec_test_100[cnt],acc_test[cnt] = innerfold(train_nodes, test_nodes)
        cnt +=1
    
    prec_mean_10 = np.round_(prec_test_10.mean(),decimals= 2)
    prec_std_10 = np.round_(prec_test_10.std(),decimals= 3)
    prec_mean_100 = np.round_(prec_test_100.mean(),decimals= 2)
    prec_std_100 = np.round_(prec_test_100.std(),decimals= 3)
    acc_mean = np.round_(acc_test.mean(),decimals= 2)
    acc_std = np.round_(acc_test.std(),decimals= 3)
    print('Precision@10 Test Mean / Std: %f $\pm$ %f' % (prec_mean_10, prec_std_10 ))
    print('Precision@100 Test Mean / Std: %f $\pm$ %f' % (prec_mean_100, prec_std_100 ))
    print('Accuracy Test Mean / Std: %f $\pm$ %f' % (acc_mean, acc_std))

