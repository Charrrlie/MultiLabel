from tqdm import tqdm
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter   

from config import DefaultConfig
from models import ImgModule, TxtModule, ClassifierModule
from utils import *

"""


"""
opt = DefaultConfig()


def train(**kwargs):
    opt.parse(kwargs)

    # Load data
    images, tags, labels = load_data(opt.data_path)
    y_dim = tags.shape[1]

    X, Y, L = split_data(images, tags, labels, opt)
    print('Loading and splitting data finished.')

    # Init module
    img_model = ImgModule(opt.bit)
    txt_model = TxtModule(y_dim, opt.bit)
    cls_model = ClassifierModule(opt.bit, opt.num_class)
   
    if opt.use_gpu:
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()
        cls_model = cls_model.cuda()

    # Data to torch tensor
    train_L = torch.from_numpy(L['train'])
    train_x = torch.from_numpy(X['train'])
    train_y = torch.from_numpy(Y['train'])

    query_L = torch.from_numpy(L['query'])
    query_x = torch.from_numpy(X['query'])
    query_y = torch.from_numpy(Y['query'])

    retrieval_L = torch.from_numpy(L['retrieval'])
    retrieval_x = torch.from_numpy(X['retrieval'])
    retrieval_y = torch.from_numpy(Y['retrieval'])

    num_train = train_x.shape[0]

    F_buffer = torch.randn(num_train, opt.bit) # tensor (num_train, bit)
    G_buffer = torch.randn(num_train, opt.bit) # tensor (num_train, bit)

    if opt.use_gpu:
        train_L = train_L.float().cuda()
        F_buffer = F_buffer.cuda()
        G_buffer = G_buffer.cuda()

    B = torch.sign(F_buffer + G_buffer) # tensor (num_train, bit)

    lr = opt.lr
    optimizer_img = Adam(img_model.parameters(), lr=lr)
    optimizer_txt = Adam(txt_model.parameters(), lr=lr)
    optimizer_cls = Adam(cls_model.parameters(), lr=lr)
    
    cls_criterion = nn.BCEWithLogitsLoss() # Multi-Label Classification(Binary Cross Entropy Loss)
    
    best_mapi2t = 0.0
    print('...training procedure starts')

    ones = torch.ones(opt.batch_size, 1)
    ones_ = torch.ones(num_train - opt.batch_size, 1)
    if opt.use_gpu:
        ones = ones.cuda()
        ones_ = ones_.cuda()

    for epoch in range(opt.max_epoch):

        # Part 1: train image net & update F & classifier
        for i in tqdm(range(num_train // opt.batch_size)):
            # Random samples
            index = np.random.permutation(num_train)
            ind = index[0: opt.batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = Variable(train_L[ind, :])   # (batch_size, num_class)
            image = Variable(train_x[ind].type(torch.float))
            if opt.use_gpu:
                image = image.cuda()
                sample_L = sample_L.cuda()
              
            cur_f = img_model(image)  # cur_f: (batch_size, bit)
            F_buffer[ind, :] = cur_f.data # update F

            pred_x = cls_model(cur_f)
            cls_x = cls_criterion(pred_x, sample_L)

            quantization_x = torch.sum(torch.pow(B[ind, :] - cur_f, 2)) / (opt.batch_size * num_train) # ||B-f||_2^F
            
            loss_x = cls_x + quantization_x 

            optimizer_img.zero_grad()
            optimizer_cls.zero_grad()
            loss_x.backward()
            optimizer_img.step()
            optimizer_cls.zero_grad()

        # Part 2: train txt net & update G & classifier
        for i in tqdm(range(num_train // opt.batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: opt.batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = Variable(train_L[ind, :])
            text = train_y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
            text = Variable(text)
            if opt.use_gpu:
                text = text.cuda()
                sample_L = sample_L.cuda()

            cur_g = txt_model(text)  # cur_g: (batch_size, bit)
            G_buffer[ind, :] = cur_g.data # update G

            pred_y = cls_model(cur_g)
            cls_y = cls_criterion(pred_y, sample_L)
            quantization_y = torch.sum(torch.pow(B[ind, :] - cur_g, 2)) / (num_train * opt.batch_size) # ||B-g||_2^F
            
            loss_y = cls_y + quantization_y 
 
            optimizer_txt.zero_grad()
            optimizer_cls.zero_grad()
            loss_y.backward()
            optimizer_txt.step()
            optimizer_cls.step()
        
        # Update B
        B = torch.sign(F_buffer + G_buffer)

        if opt.valid and True:
            mapi2t, mapt2i = evaluate(img_model, txt_model, query_x, query_y, retrieval_x, retrieval_y, query_L, retrieval_L, opt.bit)

             # save best model
            if mapi2t > best_mapi2t:
                print("best mapi2t, save model...")
                best_mapi2t = mapi2t
                txt_model.save(opt.load_txt_path)
                img_model.save(opt.load_img_path)

            print("{}".format(datetime.now()))
            print('%d...eval map: map(i->t): \033[1;32;40m%3.3f\033[0m, map(t->i): \033[1;32;40m%3.3f\033[0m' % (
                    epoch, mapi2t, mapt2i))

    print('...training procedure finish')
    mapi2t, mapt2i = evaluate(img_model, txt_model, query_x, query_y, retrieval_x, retrieval_y, query_L, retrieval_L, opt.bit)
    print('...test map: map(i->t): %3.3f, map(t->i): %3.3f' % (mapi2t, mapt2i))


def evaluate(img_model, txt_model, query_x, query_y, retrieval_x, retrieval_y, query_L, retrieval_L, bit):
    qBX = generate_image_code(img_model, query_x, opt.bit)
    qBY = generate_text_code(txt_model, query_y, opt.bit)
    rBX = generate_image_code(img_model, retrieval_x, opt.bit)
    rBY = generate_text_code(txt_model, retrieval_y, opt.bit)

    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)

    return mapi2t, mapt2i


def generate_image_code(img_model, X, bit):
    batch_size = opt.batch_size
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)

    if opt.use_gpu:
        B = B.cuda()

    for i in range(num_data // batch_size + 1):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        image = X[ind].type(torch.float)
        if image.shape[0] == 0 :
            continue
        if opt.use_gpu:
            image = image.cuda()
        cur_f = img_model(image)
        B[ind, :] = cur_f.data * 2 - 1
    B = torch.sign(B)

    return B


def generate_text_code(txt_model, Y, bit):
    batch_size = opt.batch_size
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if opt.use_gpu:
        B = B.cuda()
    for i in range(num_data // batch_size + 1):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        text = Y[ind].unsqueeze(1).unsqueeze(-1).type(torch.float)
        if opt.use_gpu:
            text = text.cuda()
        cur_g = txt_model(text)
        B[ind, :] = cur_g.data * 2 - 1
    B = torch.sign(B)

    return B


def debug(**kwargs):
    opt.parse(kwargs)

    # load data
    images, tags, labels = load_data(opt.data_path)
    y_dim = tags.shape[1]
    labels1 = np.load(opt.l1_path)

    X, Y, L, L1 = split_data(images, tags, labels, opt, labels1)

    print('...loading and splitting data finish')

    # init module
    img_model = ImgModule(opt.bit)
    txt_model = TxtModule(y_dim, opt.bit)

    if opt.use_gpu:
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()
    
    print("load trained model from file..")
    img_model.load(opt.load_img_path, use_gpu=True)
    txt_model.load(opt.load_txt_path, use_gpu=True)

    train_L = torch.from_numpy(L['train'])
    train_L1 = torch.from_numpy(L1['train'])
    train_x = torch.from_numpy(X['train'])
    train_y = torch.from_numpy(Y['train'])

    query_L = torch.from_numpy(L['query'])
    query_x = torch.from_numpy(X['query'])
    query_y = torch.from_numpy(Y['query'])

    retrieval_L = torch.from_numpy(L['retrieval'])
    retrieval_x = torch.from_numpy(X['retrieval'])
    retrieval_y = torch.from_numpy(Y['retrieval'])

    mapi2t, mapt2i = evaluate(img_model, txt_model, query_x, query_y, retrieval_x, retrieval_y, query_L, retrieval_L, opt.bit)
    print('...test map: map(i->t): %3.3f, map(t->i): %3.3f' % (mapi2t, mapt2i))


if __name__ == '__main__':
    train() 