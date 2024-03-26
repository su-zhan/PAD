import os
import gzip
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from divtype import *


def get_train_loader(logger, fold, train_ids, BATCH_SIZE, seq_len, window, overlap, pas_num):
    data_list = []
    logger.info('Begin loading fold {} training data'.format(fold))
    logger.info('trian_ids = {}'.format(train_ids))
    data_dir = '../data/train_' + str(seq_len) + '_' + str(window) + '_' + str(overlap) + '_' + str(pas_num) + '/'
    doc_tensor_dict = {}
    for qid in tqdm(train_ids,desc='load train data', ncols=80):
        file_path1 = os.path.join(data_dir, str(qid) + '_emb.pkl.gz')
        with gzip.open(file_path1,'rb') as f:
            try:
                temp_doc_tensor_dict=pickle.load(f)
            except EOFError:
                continue
        doc_tensor_dict[str(qid)] = temp_doc_tensor_dict[str(qid)]
        file_path = os.path.join(data_dir, str(qid) + '_sample.pkl.gz')
        with gzip.open(file_path,'rb') as f:
            try:
                temp_dict = pickle.load(f)
            except EOFError:
                continue
        temp_data_list = [
            (t[0], # weight
            t[1], # pos_mask
            t[2], # neg_mask
            t[3], # context_mask
            t[4], # candidate mask
            doc_tensor_dict[str(qid)][0], # X
            doc_tensor_dict[str(qid)][1], # segment
            doc_tensor_dict[str(qid)][2], # position
            doc_tensor_dict[str(qid)][3], # attn_mask
            doc_tensor_dict[str(qid)][4] # rel_feat
            ) for t in temp_dict[str(qid)]]
        data_list.extend(temp_data_list)
    train_dataset = TrainDataset(data_list)
    loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    return loader


def get_test_loader(logger, fold, test_qids, BATCH_SIZE, seq_len, window, overlap, pas_num):
    data_dir = '../data/test_' + str(seq_len) + '_' + str(window) + '_' + str(overlap) + '_' +str(pas_num) + '/'
    test_dataset = {}
    for qid in tqdm(test_qids, desc = "load test data", ncols = 80):
        file_path = os.path.join(data_dir,str(qid)+'.pkl.gz')
        with gzip.open(file_path,'rb') as f:
            try:
                temp_test_dict=pickle.load(f)
            except EOFError:
                continue
        test_dataset[str(qid)]=temp_test_dict[str(qid)]
    test_data_list = [
        (test_dataset[qid][0], # X
        test_dataset[qid][1], # segment
        test_dataset[qid][2], # position
        test_dataset[qid][3], # attn_mask
        test_dataset[qid][4], # rel_feat
        test_dataset[qid][5], # context_mask
        test_dataset[qid][6], # candidate_mask
        )
        for qid in test_qids
    ]
    evaluate_dataset=TestDataset(test_data_list)
    loader = DataLoader(
        dataset=evaluate_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return loader
