import numpy as np
from mxnet.contrib import text
import torch.utils.data as data_utils
import torch
import random
import seaborn as sns
import scipy.sparse as sp
def Bload_data(batch_size=60,num_class=18):
    #AAPD data
    X_trn = np.load("data/AAPD/X_train.npy")
    Y_trn = np.load("data/AAPD/y_train.npy")
    embed = text.embedding.CustomEmbedding('data/AAPD/word_embed.txt')
    base = Y_trn[..., :num_class]
    base1 = base.T
    base_id = []
    for j in base1:
        m = np.nonzero(j)
        base_id.extend(m[0])
    base_id = list(np.unique(base_id))
    base_val = random.sample(base_id, 1000)
    base_trn = list(set(base_id) - set(base_val))
    basey_trn = base[base_trn]
    basex_trn = X_trn[base_trn]
    base_valX = X_trn[base_val]
    base_valY=base[base_val]

    base_train = data_utils.TensorDataset(torch.from_numpy(basex_trn).type(torch.LongTensor),
                                          torch.from_numpy(basey_trn).type(torch.LongTensor))
    base_val = data_utils.TensorDataset(torch.from_numpy(base_valX).type(torch.LongTensor),
                                          torch.from_numpy(base_valY).type(torch.LongTensor))
    Btrain_loader = data_utils.DataLoader(base_train, batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
    Btest_loader = data_utils.DataLoader(base_val, batch_size, shuffle=False,drop_last=True, num_workers=4, pin_memory=True)
    return Btrain_loader, Btest_loader, embed.idx_to_vec.asnumpy()


def Nload_data(batch_size=120,sample_num=5,samp_freq=30,num_class=45):
    #AAPD data
    X_tst = np.load("data/AAPD/X_test.npy")
    Y_tst = np.load("data/AAPD/y_test.npy")
    X_trn = np.load("data/AAPD/X_train.npy")
    Y_trn = np.load("data/AAPD/y_train.npy")
    novel1=Y_trn[...,num_class:]
    novel=novel1.T
    novel_sam=[]
    for m in range(samp_freq):
        for i in novel:
            w = np.nonzero(i)
            novel_sam.append(random.sample(list(w[0]),sample_num))
    novel_x=[]
    novel_y=[]
    novelall_y=[]
    for i in novel_sam:
        novel_x.extend(X_trn[i])
        novel_y.extend(novel1[i])
        novelall_y.extend(Y_trn[i])
    novel_y=np.array(novel_y)
    novel_x=np.array(novel_x)
    novelall_Y=np.array( novelall_y)

    test_data = data_utils.TensorDataset(torch.from_numpy(X_tst).type(torch.LongTensor),
                                         torch.from_numpy(Y_tst).type(torch.LongTensor))

    novelall_data = data_utils.TensorDataset(torch.from_numpy(novel_x).type(torch.LongTensor),
                                         torch.from_numpy(novelall_Y).type(torch.LongTensor))

    novel_data = data_utils.TensorDataset(torch.from_numpy(novel_x).type(torch.LongTensor),
                                         torch.from_numpy(novel_y).type(torch.LongTensor))

    novelall_loader=data_utils.DataLoader(novelall_data,batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
    Ftest_loader = data_utils.DataLoader(test_data,batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
    novel_loader = data_utils.DataLoader(novel_data, batch_size, shuffle=False,drop_last=True, num_workers=4, pin_memory=True)
    return Ftest_loader, novel_loader,novelall_loader

def one_sample_base2avg(batch_size=30,sample_num=5,samp_freq=30,num_class=36):
    #AAPD data
    X_trn = np.load("data/AAPD/X_train.npy")
    Y_trn = np.load("data/AAPD/y_train.npy")
    base = Y_trn[..., :num_class]

    base1 = base.T

    base_id=[]
    for j in base1:
        m = np.nonzero(j)
        base_id.extend(m[0])
    base_id = list(np.unique(base_id))
    base_val = random.sample(base_id, 1000)
    base_trn = list(set(base_id) - set(base_val))
    Y_trn = base[base_trn]
    X_trn = X_trn[base_trn]

    embed = text.embedding.CustomEmbedding('data/AAPD/word_embed.txt')

    Y_trn1 = Y_trn.T
    base_sam = []
    for n in range(samp_freq):
        for i in Y_trn1:
            w = np.nonzero(i)
            base_sam.append(random.sample(list(w[0]), sample_num))
    base_x = []
    base_y = []
    for i in base_sam:
        base_x.extend(X_trn[i])
        base_y.extend(Y_trn[i])

    base_x = np.array(base_x)
    base_y = np.array(base_y)
    transf = data_utils.TensorDataset(torch.from_numpy(base_x).type(torch.LongTensor),
                                      torch.from_numpy(base_y).type(torch.LongTensor))
    base_transf = data_utils.DataLoader(transf, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return base_transf, embed.idx_to_vec.asnumpy()

    