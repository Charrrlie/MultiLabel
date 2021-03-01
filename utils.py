import h5py
import numpy as np
import torch


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))

    return distH


def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]    
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map


def calc_loss(B, F, G, Sim, gamma, eta):
    """TODO: change
    """
    theta = torch.matmul(F, G.transpose(0, 1)) / 2
    term1 = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
    term2 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
    term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
    loss = term1 + gamma * term2 + eta * term3

    return loss


def split_data(images, tags, labels, opt):
    X = {}
    X['query'] = images[0: opt.query_size]
    X['train'] = images[opt.query_size: opt.training_size + opt.query_size]
    X['retrieval'] = images[opt.query_size: opt.query_size + opt.database_size]

    Y = {}
    Y['query'] = tags[0: opt.query_size]
    Y['train'] = tags[opt.query_size: opt.training_size + opt.query_size]
    Y['retrieval'] = tags[opt.query_size: opt.query_size + opt.database_size]

    L = {}
    L['query'] = labels[0: opt.query_size]
    L['train'] = labels[opt.query_size: opt.training_size + opt.query_size]
    L['retrieval'] = labels[opt.query_size: opt.query_size + opt.database_size]

    return X, Y, L


def load_data(path):
    """ Load data(numpy) from doc.
    
    e.g. flickr-25k
    np.shape(images): (20015, 3, 224, 224), image tensor
    np.shape(texts): (20015, 1386), 1386-D BoW tensor
    np.shape(labels): (20015, 24), 24-D 0/1 tensor
    """
    file = h5py.File(path, 'r')
    
    images = file['images'][:].astype('float') 
    labels = file['LAll'][:] 
    tags = file['YAll'][:] 

    file.close()
    return images, tags, labels


def fill_l1():
    data_path = "/Users/chan/Desktop/data/多模态/MIRFLICKR-25K/FLICKR-25K.mat"
    file = h5py.File(data_path, 'r')
    labels = file['LAll'][:]
    print(labels[0])
    file.close()

    ALL = np.zeros((20015, 10))

    l1 = [4, 18, 20]
    l2 = [19]
    l3 = [12]   
    l4 = [9] 
    l5 = [8] 
    l6 = [21, 3] 
    l7 = [23, 17, 16, 10] 
    l8 = [22, 7, 14] 
    l9 = [13, 15, 11, 6, 1]
    l10 = [0, 5, 2]

    for i, l in enumerate(labels): 
        index = np.argwhere(l > 0)
        for j in index:
            if j in l1:
                ALL[i][0] = 1
            elif j in l2:
                ALL[i][1] = 1
            elif j in l3:
                ALL[i][2] = 1
            elif j in l4:
                ALL[i][3] = 1
            elif j in l5:
                ALL[i][4] = 1
            elif j in l6:
                ALL[i][5] = 1
            elif j in l7:
                ALL[i][6] = 1
            elif j in l8:
                ALL[i][7] = 1
            elif j in l9:
                ALL[i][8] = 1
            elif j in l10:
                ALL[i][9] = 1

    np.save('L1ALL', ALL)


def calc_neighbor(label1, label2, use_gpu=True):
    # calculate the similar matrix
    if use_gpu:
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
    else:
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.FloatTensor)
    return Sim


def create_semantic_labels():
    import gensim

    label_path = "./result/label.txt"
    label_list = []
    # load labels
    with open(label_path, 'r') as f:
        doc = f.readlines()
        for i in doc:
            label_list.append(i.strip())
    # load word vectors
    wv_path = "./data/glove.6B.50d.txt"
    label_vec = np.zeros((24, 50))

    model = gensim.models.KeyedVectors.load_word2vec_format(wv_path, binary=False)
    vocab_list = [word for word in model.vocab.keys()]
    word2id = {}
    vectors = np.zeros((len(vocab_list), model.vector_size))
    for i, word in enumerate(vocab_list):
        word2id[word] = i
        vectors[i] = model.wv[word]
    
    for i, label in enumerate(label_list):
        wordid = word2id[label]
        vec = vectors[wordid]
        label_vec[i] = vec
    
    # save
    np.save("./data/label_vec", label_vec)
    print(label_vec)
    

if __name__ == '__main__':  
    # Test MAP
    qB = torch.Tensor([[1, -1, 1, 1],
                       [-1, -1, -1, 1],
                       [1, 1, -1, 1],
                       [1, 1, 1, -1]])
    rB = torch.Tensor([[1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [1, 1, -1, -1],
                       [-1, 1, -1, -1],
                       [1, 1, -1, 1]])
    query_L = torch.Tensor([[0, 1, 0, 0],
                            [1, 1, 0, 0],
                            [1, 0, 0, 1],
                            [0, 1, 0, 1]])
    retrieval_L = torch.Tensor([[1, 0, 0, 1],
                                [1, 1, 0, 0],
                                [0, 1, 1, 0],
                                [0, 0, 1, 0],
                                [1, 0, 0, 0],
                                [0, 0, 1, 0]])

    map = calc_map_k(qB, rB, query_L, retrieval_L)
    print(map)
