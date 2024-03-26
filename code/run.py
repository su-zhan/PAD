# -*- coding:utf-8 -*-
import os
import torch
import random
import logging
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from PAD import PAD
from divtype import *
import evaluate as EV
import data_process as DP
from sklearn.model_selection import KFold
from torch.nn.utils import clip_grad_norm_


MAXDOC=50
REL_LEN=18
LR_list=[0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5, 9e-6, 8e-6, 7e-6, 6e-6, 5e-6, 4e-6, 3e-6, 2e-6, 1e-6]


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def decay_LR(lr):
    index = LR_list.index(lr)
    if index == -1 or lr <= 1e-6:
        return LR_list[0] 
    else:
        return LR_list[index+1]


def list_pairwise_loss(score_1, score_2, delta):
    loss = -torch.sum(delta * torch.Tensor.log(1e-8+torch.sigmoid(score_1 - score_2)))/float(score_1.shape[0])
    return loss


def run(old_best_model_list, max_metric_list, logger, BATCH_SIZE, EPOCH, LR, DROPOUT, SEQ_LEN, WINDOW, OVERLAP, PAS_NUM_PER_DOC, INTERVAL, TEST_BATCH):
    tmp_dir = '../tmp/Batch_' + str(BATCH_SIZE) + '_EPOCH_' + str(EPOCH) + '_LR_' + str(LR) + '_DROP_' + str(DROPOUT) + '_SEQ_' + str(SEQ_LEN) + '_INTERVAL_' + str(INTERVAL) + '/'
    logger.info('tmp_dir = {}'.format(tmp_dir))
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    all_qids = np.load('../data/all_qids.npy')
    logger.info('all qids:' + str(len(all_qids)))
    logger.info('numpy shuffle:' + str(all_qids))
    qd = pickle.load(open('../data/div_query.data','rb'))
    final_metrics = []
    best_model_list = []
    fold = 0

    test_qids_list=[]
    for train_ids, test_ids in KFold(5).split(all_qids):
        fold += 1
        logger.info('Fold = {}'.format(fold))
        best_model = old_best_model_list[fold-1]
        decay_flag = False if best_model == "" else True
        max_metric = max_metric_list[fold-1]

        train_ids.sort()
        test_ids.sort()
        train_qids=[str(all_qids[i]) for i in train_ids]
        test_qids=[str(all_qids[i]) for i in test_ids]
        test_qids_list.append(test_qids)

        train_data_loader = DP.get_train_loader(logger, fold, train_qids, BATCH_SIZE, SEQ_LEN, WINDOW, OVERLAP, PAS_NUM_PER_DOC)
        test_data_loader = DP.get_test_loader(logger, fold, test_qids, TEST_BATCH, SEQ_LEN, WINDOW, OVERLAP, PAS_NUM_PER_DOC)

        model = PAD(DROPOUT)
        if torch.cuda.is_available():
            model=model.cuda()
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        params = list(model.parameters())
        if fold == 1 and not decay_flag:
            logger.info('model = {}'.format(model))
            logger.info('len params = {}'.format(len(params)))
            for param in params:
                logger.info('{}'.format(param.size()))
            n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
            logger.info('* number of parameters: %d' % n_params)

        all_steps = len(train_data_loader)
        
        patience = 0
        for epoch in range(EPOCH):
            logger.info('Start Training...')
            model.train()
            for step, train_data in enumerate(tqdm(train_data_loader, desc='BATCH', ncols=80)):
                X, segment, positions, attn_mask, rel_feat, context_mask, candidate_mask, pos_mask, neg_mask, weight = train_data
                if torch.cuda.is_available():
                    X = X.cuda()
                    segment = segment.cuda()
                    positions = positions.cuda()
                    attn_mask = attn_mask.cuda()
                    rel_feat = rel_feat.cuda()
                    context_mask = context_mask.cuda()
                    candidate_mask = candidate_mask.cuda()
                    pos_mask = pos_mask.cuda()
                    neg_mask = neg_mask.cuda()
                    weight = weight.cuda()
                    
                score1, score2 = model(PAS_NUM_PER_DOC, X, segment, positions, attn_mask, rel_feat, context_mask, candidate_mask, pos_mask, neg_mask, True)
                loss = list_pairwise_loss(score1, score2, weight)

                opt.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1)
                opt.step()

                if INTERVAL > 0 and (step + 1) % (all_steps // INTERVAL) == 0:
                    model.eval()
                    avg_alpha_NDCG = 0
                    metrics = EV.evaluate_test_qids(qd, SEQ_LEN, PAS_NUM_PER_DOC, model, test_data_loader, fold, test_qids)
                    avg_alpha_NDCG = np.mean(metrics)

                    if max_metric < avg_alpha_NDCG:
                        max_metric = avg_alpha_NDCG
                        logger.info('max avg_alpha_NDCG updated: {}'.format(max_metric))
                        model_filename = '../model/INTERVAL_' + str(INTERVAL) + '_TOTAL_' + str(EPOCH) + '_FOLD_' + str(fold) + '_EPOCH_' + str(epoch) + '_LR_' + str(LR) + '_BATCH_' + str(BATCH_SIZE) + '_DROPOUT_' + str(DROPOUT) + '_' + str(SEQ_LEN) + '_alpha_NDCG_' + str(max_metric) + '.pt'
                        torch.save(model.state_dict(), model_filename)
                        logger.info('save file at: {}'.format(model_filename))
                        best_model = model_filename
                        patience = 0
                    else: 
                        patience += 1
                    if (epoch > 0 or decay_flag) and patience > 2:
                        new_lr = 0.0
                        for param_group in opt.param_groups:
                            param_group['lr'] = decay_LR(param_group['lr'])
                            new_lr = param_group['lr'] 
                        patience = 0
                        logger.info("decay lr: {}, load model: {}".format(new_lr, best_model))
                        model.load_state_dict(torch.load(best_model))
                    model.train()

            model.eval()
            metrics = EV.evaluate_test_qids(qd, SEQ_LEN, PAS_NUM_PER_DOC, model, test_data_loader, fold, test_qids)
            avg_alpha_NDCG = np.mean(metrics)
            if max_metric < avg_alpha_NDCG:
                max_metric = avg_alpha_NDCG
                logger.info('max avg_alpha_NDCG updated: {}'.format(max_metric))
                model_filename = '../model/INTERVAL_' + str(INTERVAL) + '_TOTAL_' + str(EPOCH) + '_FOLD_' + str(fold) + '_EPOCH_' + str(epoch) + '_LR_' + str(LR) + '_BATCH_' + str(BATCH_SIZE) + '_DROPOUT_' + str(DROPOUT) + '_' + str(SEQ_LEN) + '_alpha_NDCG_' + str(max_metric) + '.pt'
                torch.save(model.state_dict(), model_filename)
                logger.info('save file at: {}'.format(model_filename))
                best_model = model_filename
            if epoch == (EPOCH-1):
                final_metrics.append(max_metric)
                best_model_list.append(best_model)
    logger.info('final_metrics = {}'.format(final_metrics))
    logger.info('best_model_list = {}'.format(best_model_list))
    return best_model_list, final_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--per_gpu_batch_size', type=int, default=8,help="the single GPU batch size")
    parser.add_argument('--per_gpu_test_batch', type=int, default=10,help="the single GPU batch size in test")
    parser.add_argument('--epoch', type=int, nargs = '*', help="the training epoches")
    parser.add_argument('--lr', type=float, default=1e-3,help="Which learning rate to start with. (Default: 1e-3)")
    parser.add_argument('--dropout', type=float, default=0.5,help="the dropout rate")
    parser.add_argument('--seqlen', type=int, default=301, help="the sequence len of doc")
    parser.add_argument('--pas_num_per_doc', type=int, default=5, help="passage number per doc")
    parser.add_argument('--window', type=int, default=256, help="the window size of passage")
    parser.add_argument('--overlap', type=int, default=16, help="the overlap size of passage")
    parser.add_argument('--interval', type=int, default=50, help="the interval step for evaluating model")
    parser.add_argument('--device', type=str, default="0",help="GPU ID")
    parser.add_argument('--log_path', type=str, default="../log/train.log",help="log file path")
    
    set_seed()
    args = parser.parse_args()
    DEVICE = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count()
    args.test_batch = args.per_gpu_test_batch * torch.cuda.device_count()
    BATCH_SIZE = args.batch_size
    TEST_BATCH = args.test_batch
    LR = args.lr
    DROPOUT = args.dropout
    INTERVAL = args.interval
    PAS_NUM_PER_DOC = args.pas_num_per_doc
    SEQ_LEN = args.seqlen
    WINDOW = args.window
    OVERLAP = args.overlap
    log_path = args.log_path

    logging.basicConfig(filename=log_path, level=logging.DEBUG)  
    logger = logging.getLogger(__name__)
    logger.info('number GPU = {}'.format(torch.cuda.device_count()))
    logger.info('total_batch = {}'.format(BATCH_SIZE))
    logger.info(args)

    best_model_list = [''] * 5
    max_metric_list = [0] * 5
    epoch_list = list(args.epoch)
    for i in range(len(epoch_list)):
        logger.info('PERIOD = {}'.format(i))
        best_model_list, max_metric_list = run(best_model_list, max_metric_list, logger, BATCH_SIZE, epoch_list[i], LR, DROPOUT, SEQ_LEN, WINDOW, OVERLAP, PAS_NUM_PER_DOC, INTERVAL, TEST_BATCH)
    logger.info('done!')
