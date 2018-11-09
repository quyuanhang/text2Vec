import sys
import jieba
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np


def count_degree(frame, col):
    user = frame.columns[col]
    user_degree_series = frame.iloc[:, col]
    user_degree_frame = pd.DataFrame(user_degree_series.value_counts())
    user_degree_frame.columns = ['degree']
    user_degree_frame = pd.merge(frame, user_degree_frame,
                                 left_on=user, right_index=True)
    return user_degree_frame


def filter_old(frame, N=0, M=100000):
    frame = count_degree(frame, 0)
    frame = count_degree(frame, 1)
    old_frame = frame[(frame['degree_x'] >= N) & (frame['degree_y'] >= N)]
    # print('rest users', len(set(old_frame.iloc[:, 0])))
    # print('rest items', len(set(old_frame.iloc[:, 1])))
    # print('rest matches', len(old_frame))
    return old_frame.iloc[:, :2]


def iter_filter_old(frame, N=10, M=5, step=100):
    for i in range(step):
        frame = filter_old(frame.iloc[:, :2], N, M)
        frame = count_degree(frame, 0)
        frame = count_degree(frame, 1)
        # print(frame.head())
        if frame['degree_x'].min() >= N and frame['degree_y'].min() >= M:
            break
    return frame.iloc[:, :2]


def train_test_split(df):
    train = list()
    test = list()

    print('all data')
    print(len(df))
    df.index = df['job']

    print('spliting ...')
    for job in tqdm(set(df['job'])):
        job_action = df.loc[job].values
        train.extend(job_action[:-1])
        test.append(job_action[-1])

    train = pd.DataFrame(train, columns=['job', 'geek'])
    test_set = set([(job, geek) for job, geek in test])
    test = pd.DataFrame(test, columns=['job', 'geek'])

    return train, test, test_set


def build_user_index(train):
    job_dict = {k:v for v, k in enumerate(train['job'].drop_duplicates())}
    geek_dict = {k:v for v, k in enumerate(train['geek'].drop_duplicates())}
    print('n_job n_geek')
    print(len(job_dict), len(geek_dict))
    return job_dict, geek_dict


def convert_data(frame, job_dict, geek_dict, filter_set={}):
    data = list()
    for job, geek in tqdm(frame.values):
       if (job, geek) in filter_set:
           continue
       if job in job_dict and geek in geek_dict:
           data.append([job_dict[job], geek_dict[geek]])
    frame = pd.DataFrame(data, columns=['job', 'geek'])
    return frame


def neg_sample(train, test, all_geek, n_neg=101):
    train.index = train['job']
    test_samples = list()
    print('negative sampling ...')
    for job, geek in tqdm(test.values):
        positive = set(train.loc[job])
        positive.add(geek)
        negative = list()
        while len(negative) < n_neg:
            requeir = int((n_neg - len(negative)) * 1.2)
            samples = np.random.randint(0, all_geek, requeir)
            samples = [x for x in samples if x not in positive]
            negative.extend(samples)
        negative = negative[:n_neg]
        negative[0] = job
        negative[1] = geek
        test_samples.append(negative)
    test_frame = pd.DataFrame(test_samples)
    return test_frame


def train_frame_plus(train_frame, job_add, geek_add, job_dict, geek_dict, test_set):
    print('interview num', len(train_frame))
    print('converting job index ...')
    job_add = convert_data(job_add, job_dict, geek_dict, test_set)
    job_add['rating'] = 2
    print('job posi num', len(job_add))
    train_frame = pd.concat((train_frame, job_add), sort=False)
    print('converting geek index ...')
    geek_add = convert_data(geek_add, job_dict, geek_dict, test_set)
    geek_add['rating'] = 1
    print('geek posi num', len(geek_add))
    train_frame = pd.concat((train_frame, geek_add), sort=False)
    return train_frame


def data_format(d):
    if type(d) == float:
        return str(int(d))
    else:
        return str(d)


def profile(path, str_id_dict):
    datas = [''] * len(str_id_dict)
    with open(path) as f:
        for line in tqdm(f):
            data = line.split(SEP)
            strid = data[0]
            if strid not in str_id_dict:
                continue
            data[-1] = ' '.join(jieba.cut(data[-1]))
            # prof = '\t'.join(data[1:])
            prof = ' '.join(data[1:])
            prof = prof.replace('\n', '')
            idx = str_id_dict[strid]
            datas[idx] = prof
    # for k, v in str_id_dict.items():
    #     datas[v] = k + '\t' + datas[v]
    return datas


def profile_format(inpath, outpath, job_dict, geek_dict):
    job_profile_path = '{}.profile.job'.format(inpath)
    job_profile = profile(job_profile_path, job_dict)
    print(len(job_profile))
    with open('./data/{}.profile.job'.format(outpath), 'w') as f:
        f.write('\n'.join(job_profile))
    # with open('./data/{}.position.job'.format(outpath), 'w') as f:
    #     f.write('\n'.join(job_position))

    geek_profile_path = '{}.profile.geek'.format(inpath)
    geek_profile = profile(geek_profile_path, geek_dict)
    print(len(geek_profile))
    with open('./data/{}.profile.geek'.format(outpath), 'w') as f:
        f.write('\n'.join(geek_profile))

    with open('./data/{}.profile'.format(outpath), 'w') as f:
        f.write('\n'.join((job_profile + geek_profile)))
    return


def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--datain', nargs='?', default='ml-1m')
    parser.add_argument('--dataout', default='interview')
    parser.add_argument('--t', type=int, default=5)
    parser.add_argument('--sep', default='\001')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    T = args.t
    DATA_IN = args.datain
    DATA_OUT = args.dataout
    SEP = args.sep

    # read data
    df = pd.read_table('{}.train'.format(DATA_IN), header=None, sep=SEP).dropna().applymap(data_format)
    df.columns = ['job', 'geek']

    df = iter_filter_old(df, T, T)
    train_frame, test_frame, test_set = train_test_split(df)
    job_dict, geek_dict = build_user_index(train_frame)
    print('converting id to int ...')
    train_frame = convert_data(train_frame, job_dict, geek_dict)
    train_frame['rating'] = 3
    test_frame = convert_data(test_frame, job_dict, geek_dict)
    test_frame = neg_sample(train_frame, test_frame, len(geek_dict))

    train_frame.to_csv('data/{}.train'.format(DATA_OUT), index=False, header=False, sep='\t')
    test_frame.to_csv('data/{}.test'.format(DATA_OUT), index=False, header=False, sep='\t')

    profile_format(DATA_IN, DATA_OUT, job_dict, geek_dict)

    # print('formating addfriend data ...')
    # job_add = pd.read_table('{}_job.txt'.format(DATA_IN), header=None).dropna().applymap(data_format)
    # geek_add = pd.read_table('{}_geek.txt'.format(DATA_IN), header=None).dropna().applymap(data_format)
    # print('concating addfriend data ...')
    # train_frame = train_frame_plus(train_frame, job_add, geek_add, job_dict, geek_dict, test_set)
    #
    # train_frame.to_csv('data/{}.multi.train'.format(DATA_OUT), index=False, header=False, sep='\t')
