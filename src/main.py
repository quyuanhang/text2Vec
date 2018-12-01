#coding=utf-8
import sys
import argparse
import torch
from torch import LongTensor
# from models import MF, GMF, MLP
from MLP import MLP
from GMF import GMF
from MTL import MTL
from MF import MF
from CML import CML
from evaluate import evaluate
from dataSet import (load_train, load_test, get_train_instances,
                     get_train_instances_multi, get_train_instances_pairwise,
                     load_word_emb, load_profile)


def train_model(model, n_epoch, train_instance, test_data, learning_rate, pairwise=False, profile=False):
    # 定义loss和optimizer
    predictions = model.predict(test_data, profile)
    hr, ndcg, auc = evaluate(predictions, TOP_K)
    print('init')
    print('hit ratio: {:.6f} NDCG: {:.6f}, AUC: {:.6f}'.format(hr, ndcg, auc))

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=REG)
    for epoch in range(n_epoch):
        print('epoch {} \n'.format(epoch + 1) + '*' * 10)
        running_loss = 0.0
        for i, sample in enumerate(train_instance, 1):
            if pairwise:
                batch_loss = model.batch_fit_pairwise(model, optimizer, sample, profile)
            else:
                batch_loss = model.batch_fit(model, optimizer, sample, profile)
            running_loss += batch_loss
            if i % (len(train_instance) // 5) == 0:
                print('[{}/{}] Loss: {:.6f}'.format(
                    epoch + 1, n_epoch, running_loss / (BATCH_SIZE * i)))
        predictions = model.predict(test_data, profile)
        hr, ndcg, auc = evaluate(predictions, TOP_K)
        print('hit ratio: {:.6f} NDCG: {:.6f}, AUC: {:.6f}'.format(hr, ndcg, auc))


def parse_args():
    parser = argparse.ArgumentParser(description="Run .")
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--pre_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--reg', type=float, default=0,
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--model', default='MF')
    parser.add_argument('--pairwise', type=bool, default=False)
    parser.add_argument('--bias', type=bool, default=True)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--n_sample', type=int, default=0)
    parser.add_argument('--attention', type=bool, default=False)
    parser.add_argument('--norm', type=bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':

    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        print('using GPU')

    args = parse_args()
    TRAIN_FILE_PATH = 'data/{}.train'.format(args.dataset)
    MULTI_TRAIN_FILE_PATH = 'data/{}.multi.train'.format(args.dataset)
    TEST_FILE_PATH = 'data/{}.test'.format(args.dataset)
    BATCH_SIZE = args.batch_size
    N_EPOCH = args.epochs
    N_PRE_EPOCH = args.pre_epochs
    LERANING_RATE = args.lr
    REG = args.reg
    NEG_SAMPLE = args.num_neg
    TOP_K = args.topk
    MODEL = args.model
    PAIR = args.pairwise
    WORD_EMB_PATH = 'data/{}.word_emb'.format(args.dataset)
    GEEK_PROFILE = 'data/{}.profile.geek'.format(args.dataset)
    JOB_PROFILE = 'data/{}.profile.job'.format(args.dataset)

    train_data = load_train(TRAIN_FILE_PATH)
    test_data = load_test(TEST_FILE_PATH, n_sample=args.n_sample)

    train_instance = get_train_instances(train_data, NEG_SAMPLE, BATCH_SIZE)
    word_dict, word_emb = load_word_emb(WORD_EMB_PATH)
    position_dict, job_position, job_profile = load_profile(JOB_PROFILE, word_dict, position=True)
    geek_profile = load_profile(GEEK_PROFILE, word_dict)
    profile = {'job': LongTensor(job_profile),
               'geek': LongTensor(geek_profile),
               'position': LongTensor(job_position)}

    if MODEL == 'MLP':
        model = MLP(
            n_word=len(word_dict),
            n_positin=len(position_dict),
            shape=train_data.shape,
            emb_dim=args.dim,
            with_bias=args.bias,
            pretrain_emb=word_emb,
            weight=args.attention,
            norm=args.norm
        )
    else:
        print('no model selected')
        sys.exit(0)

    if USE_GPU:
        model.cuda()

    train_model(model, N_EPOCH, train_instance, test_data, LERANING_RATE, PAIR, profile)
    predictions = model.predict(test_data, profile)
    hr, ndcg, auc = evaluate(predictions, TOP_K)
    print('hit ratio: {:.6f} NDCG: {:.6f}, AUC: {:.6f}'.format(hr, ndcg, auc))

