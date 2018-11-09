import re
import jieba
import torch
import numpy as np
import gzip
from torch.autograd import  Variable
import torch
from torch.autograd import Variable
import  torch.nn as nn
import torch.nn.functional as f
import random
from logger import Logger
import tensorflow
from sklearn.metrics import roc_auc_score
import os

sen_maxl = 50
doc_maxl = 20
batch_size = 40
sample_1 = ['本人 已有 三年 相关 工作 经验','data 进行 行数 数据 数据处理 处理']
sample_2 = ['Boot 快速 启动   简化 配置   快速 开发','Boot 快速 启动   简化 配置   快速 开发','我 的 贡献']
batch = [['本人 已有 三年 相关 工作 经验','data 进行 行数 数据 数据处理 处理'],['Boot 快速 启动   简化 配置   快速 开发','Boot 快速 启动   简化 配置   快速 开发','我 的 贡献']]
training_step = 1
vocab_size = 80000
emb_dim = 100
sen_max_length = 20
hidden_dim = 100
EPOCH = 50
step_per_epoch = 500
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class prepro(object):
    def __init__(self, path, max_sent_word, turn_num, dataset='train'):
        self.path = path
        self.max_word = max_sent_word
        self.turn = turn_num
        self.dataset = dataset


    def _get_data(self):
        with gzip.open(self.path) as f:
            data=[]
            dialog=[]
            for line in f.readlines():
                line=line.rstrip()
          #if type(line)!=type('a'):
          #    line=bytes.decode(line)
                if line.startswith('['):
                    data.append(line)
                    line=line.split()
                    dialog.append([line[0],line[1],line[2]," ".join(line[3:])])
        return dialog

    def _send_utter_adr(self, dialog):
        tokenizer = WordPunctTokenizer()
        regexp = re.compile(r'^[A-Za-z0-9]+$')
        context = []
        for line in dialog:
            ut = tokenizer.tokenize(line[3])
            word_num = 0
            sentence = []
            for word in ut:
                if word_num < self.max_word and regexp.search(word) is not None:
                    sentence.append(word.lower())
                    word_num += 1
            context.append([line[1], ' '.join(sentence), line[2]])
        return context


    def _get_sample(self, data):
        sample_data = []
        word_vocub = []
        sample = data[:self.turn]
        for i in range(self.turn, len(data)):
            if data[i][-1] != '-':
                sample = data[i - self.turn + 1:i + 1]
                gets=self.is_sample(sample)
                if gets:
                    sample_data.append(sample)
                    for turn in sample:
                        word_vocub.extend(turn[1].split())
        word_vocub = list(set(word_vocub))
        print("%d words in %s vocub" % (len(word_vocub), self.dataset))
        return sample_data, word_vocub

    def prepro(self):
        dialog = self._get_data()
        context = self._send_utter_adr(dialog)
        sample_data, word_vocub = self._get_sample(context)
       # dt = open('%s_sample_data.txt' % self.dataset, 'w')
       # for i in range(len(sample_data)):
       #     dt.write('%s\t' % str(sample_data[i]))
        print("%s sample saved, %d samples" % (self.dataset, len(sample_data)))
        return sample_data, word_vocub


def get_glove(path):
    word_embedding_dim = 100
    with open(path) as g:
        word_dic = {line.split()[0]: np.asarray(line.split()[1:], dtype='float') for line in g}
        dt = np.array(list(word_dic.values()))
        dt = dt.reshape(-1, word_embedding_dim)
        #word_dic['unk'] = np.zeros([word_embedding_dim, ])
        word_dic['unk'] = np.mean(dt, axis=0)


        #word_dic['unk'] = np.zeros([word_embedding_dim, ])
        word_dic['pad'] = np.zeros([word_embedding_dim, ])
        '''
        print(np.array(word_dic.values()))
        word_dic['unk'] = np.mean(word_dic.values(), axis=0)
        word_dic['pad'] = np.zeros([word_embedding_dim, ])
        '''
    return word_dic




def get_embedding_dict(word_vocub, word_dic):
    word_embedding_dim = 100
    word_embedding_dict = {}
    for word in word_vocub:
        try:
            word_embedding_dict[word] = word_dic[word]
        except:
            pass
    word_embedding_dict['unk'] = np.mean(word_embedding_dict.values(), axis=0)
    word_embedding_dict['pad'] = np.zeros([word_embedding_dim, ])
    return word_embedding_dict



class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.hidden_dim = hidden_dim
        self.cnn1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                            out_channels=self.hidden_dim,
                                            kernel_size=(5, self.hidden_dim),
                                            stride=1,
                                            padding=(2, 0)),
                                  nn.ReLU(),
                                  )
        self.cnn2 = nn.Sequential(nn.Conv2d(in_channels=1,
                                            out_channels=self.hidden_dim,
                                            kernel_size=(3, self.hidden_dim),
                                            stride=1,
                                            padding=(1, 0)),
                                  nn.ReLU(),
                                  )

    def forward(self, word_emb3d):
        sen_hidden = self.cnn1(word_emb3d)
        sen_hidden = sen_hidden.permute(0, 3, 2, 1)
        sen_hidden = self.cnn2(sen_hidden)

        return sen_hidden

class Classfier(nn.Module):
    def __init__(self):
        super(Classfier, self).__init__()
        self.hidden_dim = hidden_dim
        self.Bil = nn.Bilinear(in1_features=self.hidden_dim, in2_features=self.hidden_dim, out_features=self.hidden_dim)
        self.MLP = nn.Sequential(nn.Sigmoid(),
                                 nn.Linear(in_features=hidden_dim, out_features=1),
                                 nn.Sigmoid())
    def forward(self, geek, job):
        mix = self.Bil(geek, job)
        match_score = self.MLP(mix)
        return match_score


class Encoder(nn.Module):
    def __init__(self, job=True):
        super(Encoder, self).__init__()
        self.total_word = vocab_size + 2
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(self.total_word, self.hidden_dim, padding_idx=0)
        self.job = job
        self.enc = Cnn()
    def forward(self, batch):
        batch_inputs, batch_len = self.batch_to_input(batch)
        batch_embedding = self.embedding(batch_inputs)

        batch_sen = self.enc(batch_embedding.unsqueeze(dim=1))
        batch_sen = f.max_pool2d(batch_sen, kernel_size=(sen_maxl,1)).view(-1, self.hidden_dim)

        if self.job:
            batch_doc = self.job_pool(batch_sen, batch_len)
        else:
            batch_doc = self.geek_pool(batch_sen, batch_len)
        return batch_doc

    def batch_to_input(self, batch):
        batch_inputs = []
        batch_length = []
        for sample in batch:
            batch_length.append(len(sample))
            for utterance in sample:
                batch_inputs.append(utterance)
        ##################batch_inputs = Variable(torch.LongTensor(batch_inputs), requires_grad=False).cuda()##############
        batch_inputs = Variable(torch.LongTensor(batch_inputs), requires_grad=False).cuda()

        return batch_inputs, batch_length
    def job_pool(self, inputs, length):
        batch_doc = []
        start = 0

        for i in range(len(length)):
            end = start + length[i]
            batch_doc.append(torch.max(inputs[start:end],0)[0])
            start = end
	    #batch_doc.append(f.max_pool2d(input = inputs[start:end], kernel_size = (length[i],1)).squeeze(0))
        batch_doc = torch.stack(batch_doc)
        return batch_doc

    def geek_pool(self, inputs, length):
        batch_doc = []
        start = 0

        for i in range(len(length)):
            end = start + length[i]
            batch_doc.append(torch.mean(inputs[start:end], 0))
            # batch_doc.append(f.max_pool2d(input = inputs[start:end], kernel_size = (length[i],1)).squeeze(0))
        batch_doc = torch.stack(batch_doc)
        return batch_doc

class GJ_net(nn.Module):
    def __init__(self):
        super(GJ_net, self).__init__()
        self.J_enc = Encoder(job=True)
        self.G_enc = Encoder(job=False)
        self.G_cla = Classfier()
        self.J_cla = Classfier()
        self.M_cla = Classfier()
    def forward(self, g_0_g, g_0_j, g_1_g, g_1_j, g_2_g, g_2_j, j_0_g, j_0_j, j_1_g, j_1_j, j_2_g, j_2_j, train=True):
        '''
        g_0_g geek未点击的job pair中的geek   第一位表示哪端主动action，中间一位数表示未点击，开聊未面试，面试三种状态的标志，最后一位数字表示这个向量是geek还是job
        '''

        g_0_g = self.G_enc(g_0_g)

        g_1_g = self.G_enc(g_1_g)
        g_2_g = self.G_enc(g_2_g)
        j_0_g = self.G_enc(j_0_g)
        j_1_g = self.G_enc(j_1_g)
        j_2_g = self.G_enc(j_2_g)
        g_0_j = self.J_enc(g_0_j)
        g_1_j = self.J_enc(g_1_j)
        g_2_j = self.J_enc(g_2_j)
        j_0_j = self.J_enc(j_0_j)
        j_1_j = self.J_enc(j_1_j)
        j_2_j = self.J_enc(j_2_j)

        '''
        probablity for job side
        '''

        '''
        前两个字母表示哪个分类器算的概率（分数），中间一个字母表示哪一方主动action，最后一位数表示真实的状态是怎么样
        '''
        pg_g_0 = self.G_cla(g_0_g, g_0_j)
        pm_g_0 = self.M_cla(g_0_g, g_0_j)

        pg_g_1 = self.G_cla(g_1_g, g_1_j)
        pm_g_1 = self.M_cla(g_1_g, g_1_j)
        pj_g_1 = self.J_cla(g_1_g, g_1_j)

        pg_g_2 = self.G_cla(g_2_g, g_2_j)
        pm_g_2 = self.M_cla(g_2_g, g_2_j)
        pj_g_2 = self.J_cla(g_2_g, g_2_j)

        '''
        probablity for job side
        '''
        pj_j_0 = self.J_cla(j_0_g, j_0_j)  ## 2 * batch_size
        pm_j_0 = self.M_cla(j_0_g, j_0_j)

        pg_j_1 = self.G_cla(j_1_g, j_1_j)
        pm_j_1 = self.M_cla(j_1_g, j_1_j)
        pj_j_1 = self.J_cla(j_1_g, j_1_j)

        pg_j_2 = self.G_cla(j_2_g, j_2_j)
        pm_j_2 = self.M_cla(j_2_g, j_2_j)
        pj_j_2 = self.J_cla(j_2_g, j_2_j)

        if not train:
            geek_intent = [pg_g_0, pg_g_1, pg_g_2]
            match = [pm_g_1, pm_j_1, pm_g_2, pm_j_2]
            job_intent = [pj_j_0, pj_j_1, pj_j_2]
            return geek_intent, match, job_intent

        else:

            loss_g = - ( torch.mean(torch.log(1 - pg_g_0)) + (torch.mean(torch.log(pg_g_1)) + torch.mean(torch.log(pg_g_2))) / 2 )

            loss_j = - (torch.mean(torch.log(1 - pj_j_0)) + (torch.mean(torch.log(pj_j_1)) + torch.mean(torch.log(pj_j_2))) / 2 )

            loss_m = -(torch.mean(torch.log(1 - pm_g_1)) + torch.mean(torch.log(1 - pm_j_1)) + torch.mean(torch.log(pm_g_2)) + torch.mean(torch.log(pm_j_2)))

            loss_g3 = self.cal_c3_loss(pg_g_0, pm_g_0, 0) + self.cal_c3_loss(pg_g_1, pm_g_1, 1) + self.cal_c3_loss(pg_g_2, pm_g_2, 2)

            loss_j3 = self.cal_c3_loss(pj_j_0, pm_j_0, 0) + self.cal_c3_loss(pj_j_1, pm_j_1, 1) + self.cal_c3_loss(pj_j_2, pm_j_2, 2)

            loss_m4 = self.cal_c4_loss(pg_g_1, pm_g_1, pj_g_1, 0) + self.cal_c4_loss(pg_g_2, pm_g_2, pj_g_2, 1) + self.cal_c4_loss(pg_j_1, pm_j_1, pj_j_1, 2) + self.cal_c4_loss(pg_j_2, pm_j_2, pj_j_2, 3)
            Loss = loss_g + loss_j +loss_m
            #Loss = loss_g + loss_j +loss_m + loss_g3 + loss_j3 + loss_m4
            return Loss, loss_g, loss_j, loss_m, loss_g3, loss_j3, loss_m4
    def cal_c3_loss(self, p, p_m, index):
        p_0 = 1-p
        p_1 = p * p_m
        p_2 = p * (1-p_m)
        p_final = torch.stack([p_0,p_1,p_2], dim=1)
        p_final = f.softmax(p_final)
        p_result = p_final[:,index]
        loss = -torch.mean(torch.log(p_result))
        return loss

    def cal_c4_loss(self, p_g, p_m, p_j, index):
        p_0 = p_g * (1-p_m)
        p_1 = p_g * p_m
        p_2 = p_j * (1-p_m)
        p_3 = p_j * (p_m)
        p_final = torch.stack([p_0,p_1,p_2,p_3], dim=1)
        p_final = f.softmax(p_final)
        p_result = p_final[:,index]
        loss = -torch.mean(torch.log(p_result))
        return  loss





def get_emb_para(emb_file, id2word_file):
    word_embedding_dim = 100
    with open(emb_file) as g:
        word_dic = {line.split()[0]: np.asarray(line.split()[1:], dtype='float') for line in g}
        dt = np.array(list(word_dic.values()))
        dt = dt.reshape(-1, word_embedding_dim)
        word_dic['unk'] = np.mean(dt, axis=0)
        #word_dic['unk'] = np.zeros([word_embedding_dim, ])
        # word_dic['unk'] = np.zeros([word_embedding_dim, ])
        word_dic['padd'] = np.zeros([word_embedding_dim, ])
    id2word = torch.load(id2word_file)
    word_emb_dt = []
    for key in id2word.keys():
        word = id2word[key]

        word_emb_dt.append(word_dic.get(word, word_dic['unk']))
    word_emb_dt = torch.FloatTensor(word_emb_dt)
    return word_emb_dt
def calculate_para_num(net):
    count = 0
    params = list(net.parameters())
    for layer in params:
        count_layer = 1
        for dim in layer.size():
            count_layer *= dim
        count += count_layer
    print(count)

def zero_count(dt):
    count = 0
    for l in dt:
        if len(l) == 0:
            count += 1
    print(count, len(dt))

def calculate_para_num(net):
    count = 0
    params = list(net.parameters())
    for layer in params:
        count_layer = 1
        for dim in layer.size():
            count_layer *= dim
        count += count_layer
    print(count)

#word_emb_para = get_emb_para ('./wordemb1030.txt', 'id2word.pkl')
dt_dir = './big_dt'
Logger = Logger('./tfboard/')
dt = torch.load(dt_dir + '/' +'dt.pkl')
word_emb_dt = get_emb_para (dt_dir+'/wordemb.txt', dt_dir+'/id2word.pkl')

##################cuad#########################
#jd_net = GJ_net().cuda()
gj_net = GJ_net().cuda()
gj_net.J_enc.embedding.weight.data.copy_(word_emb_dt)
gj_net.G_enc.embedding.weight.data.copy_(word_emb_dt)

calculate_para_num(gj_net)
#gj_net.embedding.weight.data.copy_(word_emb_para)
optimizer = torch.optim.Adagrad(gj_net.parameters(), lr=1e-2)

class Trainer():
    def __init__(self, gj_net, optimizer, all_dt):
        self.gj_net = gj_net
        self.optimizer = optimizer
        g_0_g, g_0_j, g_1_g, g_1_j, g_2_g, g_2_j, j_0_g, j_0_j, j_1_g, j_1_j, j_2_g, j_2_j = all_dt.values()
        self.g_0_g_train, self.g_0_g_valid, g_0_g_test = self.divide_dataset(g_0_g, nega=True)
        self.g_0_j_train, self.g_0_j_valid, g_0_j_test = self.divide_dataset(g_0_j, nega=True)
        self.g_1_g_train, self.g_1_g_valid, g_1_g_test = self.divide_dataset(g_1_g)
        self.g_1_j_train, self.g_1_j_valid, g_1_j_test = self.divide_dataset(g_1_j)
        self.g_2_g_train, self.g_2_g_valid, g_2_g_test = self.divide_dataset(g_2_g)
        self.g_2_j_train, self.g_2_j_valid, g_2_j_test = self.divide_dataset(g_2_j)

        self.j_0_g_train, self.j_0_g_valid, j_0_g_test = self.divide_dataset(j_0_g, nega=True)
        self.j_0_j_train, self.j_0_j_valid, j_0_j_test = self.divide_dataset(j_0_j, nega=True)
        self.j_1_g_train, self.j_1_g_valid, j_1_g_test = self.divide_dataset(j_1_g)
        self.j_1_j_train, self.j_1_j_valid, j_1_j_test = self.divide_dataset(j_1_j)
        self.j_2_g_train, self.j_2_g_valid, j_2_g_test = self.divide_dataset(j_2_g)
        self.j_2_j_train, self.j_2_j_valid, j_2_j_test = self.divide_dataset(j_2_j)
    def train(self):
        g_0_g_b, g_0_j_b = self.get_random_batch(self.g_0_g_train, self.g_0_j_train)
        g_1_g_b, g_1_j_b = self.get_random_batch(self.g_1_g_train, self.g_1_j_train)
        g_2_g_b, g_2_j_b = self.get_random_batch(self.g_2_g_train, self.g_2_j_train)
        j_0_g_b, j_0_j_b = self.get_random_batch(self.j_0_g_train, self.j_0_j_train)
        j_1_g_b, j_1_j_b = self.get_random_batch(self.j_1_g_train, self.j_1_j_train)
        j_2_g_b, j_2_j_b = self.get_random_batch(self.j_2_g_train, self.j_2_j_train)
        loss, loss_g, loss_j, loss_m, loss_g3, loss_j3, loss_m4 = self.gj_net(g_0_g_b, g_0_j_b, g_1_g_b, g_1_j_b, g_2_g_b, g_2_j_b, j_0_g_b, j_0_j_b, j_1_g_b, j_1_j_b, j_2_g_b, j_2_j_b)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, loss_g, loss_j, loss_m, loss_g3, loss_j3, loss_m4


    def valid(self):
        self.gj_net.eval()
        valid_step = int(1000 / batch_size)
        valid_total_num = valid_step * batch_size
        geek_pos_score = np.zeros(1,dtype=np.float)
        geek_nega_score = np.zeros(1,dtype=np.float)
        match_pos_score = np.zeros(1,dtype=np.float)
        match_nega_score = np.zeros(1,dtype=np.float)
        job_pos_score = np.zeros(1,dtype=np.float)
        job_nega_score = np.zeros(1,dtype=np.float)

        for v_step in range (valid_step):
            print(v_step)
            g_0_g_b, g_0_j_b = self.get_ordered_batch(self.g_0_g_valid, self.g_0_j_valid, v_step, nega=True)
            g_1_g_b, g_1_j_b = self.get_ordered_batch(self.g_1_g_valid, self.g_1_j_valid, v_step)
            g_2_g_b, g_2_j_b = self.get_ordered_batch(self.g_2_g_valid, self.g_2_j_valid, v_step)

            j_0_g_b, j_0_j_b = self.get_ordered_batch(self.j_0_g_valid, self.j_0_j_valid, v_step, nega=True)
            j_1_g_b, j_1_j_b = self.get_ordered_batch(self.j_1_g_valid, self.j_1_j_valid, v_step)
            j_2_g_b, j_2_j_b = self.get_ordered_batch(self.j_2_g_valid, self.j_2_j_valid, v_step)
            geek_intent, match, job_intent = self.gj_net(g_0_g_b, g_0_j_b, g_1_g_b, g_1_j_b, g_2_g_b, g_2_j_b, j_0_g_b, j_0_j_b, j_1_g_b, j_1_j_b, j_2_g_b, j_2_j_b, train=False)


            geek_pos_score = np.hstack((geek_pos_score, geek_intent[1][:,0].cpu().data.numpy()))
            geek_pos_score = np.hstack((geek_pos_score, geek_intent[2][:,0].cpu().data.numpy()))
            geek_nega_score = np.hstack((geek_nega_score, geek_intent[0][:,0].cpu().data.numpy()))

            #print(geek_nega_score.shape)
            match_pos_score = np.hstack((match_pos_score, match[2][:,0].cpu().data.numpy()))
            match_pos_score = np.hstack((match_pos_score, match[3][:,0].cpu().data.numpy()))
            match_nega_score = np.hstack((match_nega_score, match[0][:,0].cpu().data.numpy()))
            match_nega_score = np.hstack((match_nega_score, match[1][:,0].cpu().data.numpy()))
            job_pos_score = np.hstack((job_pos_score, job_intent[1][:,0].cpu().data.numpy()))
            job_pos_score = np.hstack((job_pos_score, job_intent[2][:,0].cpu().data.numpy()))
            job_nega_score = np.hstack((job_nega_score, job_intent[0][:,0].cpu().data.numpy()))
        pos_label = np.ones(2*valid_total_num, dtype=np.int16)
        neg_label = np.zeros(2*valid_total_num, dtype=np.int16)
        valid_label = np.hstack((pos_label, neg_label))
        
        geek_score = np.hstack((geek_pos_score[1:], geek_nega_score[1:]))
        match_score = np.hstack((match_pos_score[1:], match_nega_score[1:]))
        print('pos',match_pos_score[1:],'neg',match_nega_score[1:])
        job_score = np.hstack((job_pos_score[1:], job_nega_score[1:]))

        geek_auc = roc_auc_score(valid_label, geek_score)
        match_auc = roc_auc_score(valid_label, match_score)
        job_auc = roc_auc_score(valid_label, job_score)

        loss_geek = - (np.mean(np.log(geek_pos_score[1:])) + np.mean(np.log(1-geek_nega_score[1:])))
        loss_match = - (np.mean(np.log(match_pos_score[1:])) + np.mean(np.log(1-match_nega_score[1:])))
        loss_job = - (np.mean(np.log(job_pos_score[1:])) + np.mean(np.log(1 - job_nega_score[1:])))

        print('geek_auc', geek_auc, 'match_auc', match_auc, 'job_auc', job_auc, 'loss_geek', loss_geek, 'loss_match', loss_match, 'loss_job', loss_job)
        self.gj_net.train()
        return geek_auc, match_auc, job_auc, loss_geek, loss_match, loss_job



    def get_random_batch(self, job_dt, geek_dt):
        assert len(job_dt) == len(geek_dt)
        length = len(job_dt)
        batch_index = random.sample(range(0, length), batch_size)
        job_batch = []
        geek_batch = []
        for index in batch_index:
            job_batch.append(job_dt[index])
            geek_batch.append(geek_dt[index])
        return job_batch, geek_batch

    def get_ordered_batch(self, job_dt, geek_dt, order, nega=False):
        assert len(job_dt) == len(geek_dt)

        if nega:
            return job_dt[order * 2*batch_size:(order +1) * 2*batch_size], geek_dt[order * 2*batch_size:(order + 1)  * 2*batch_size]
        else:
            return job_dt[order * batch_size:(order + 1) * batch_size], geek_dt[order * batch_size:(order + 1) * batch_size]

    def divide_dataset(self, dt, nega=False):
        if not nega:
            return dt[:-2000], dt[-2000:-1000], dt[-1000:]
        else:
            return dt[:-4000], dt[-4000:-2000], dt[-2000:]

trainer = Trainer(gj_net, optimizer, dt)

for epoch in range(EPOCH):
    for step in range (step_per_epoch):
        total_step = epoch * step_per_epoch + step

        loss, loss_g, loss_j, loss_m, loss_g3, loss_j3, loss_m4= trainer.train()
        Logger.scalar_summary('loss',loss,total_step)
        if (step+1) % 10 == 0:
            #print('epoch',epoch,'step',step,'train_loss',loss.cpu().data.numpy(),'pos_score',pos_score.cpu().data.numpy(),'neg_score',neg_score.cpu().data.numpy())
            print('epoch', epoch, 'step', step, 'train_loss', loss.cpu().data.numpy(), 'loss_g', loss_g.cpu().data.numpy(), 'loss_j', loss_j.cpu().data.numpy(), 'loss_m', loss_m.cpu().data.numpy(), 'loss_g3', loss_g3.cpu().data.numpy(), 'loss_j3', loss_j3.cpu().data.numpy(), 'loss_m4', loss_m4.cpu().data.numpy() )
    geek_auc, match_auc, job_auc, loss_geek, loss_match, loss_job = trainer.valid()
    Logger.scalar_summary('geek_auc', geek_auc, total_step)
    Logger.scalar_summary('match_auc', match_auc, total_step)
    Logger.scalar_summary('job_auc', job_auc, total_step)
    Logger.scalar_summary('loss_geek', loss_geek, total_step)
    Logger.scalar_summary('loss_match', loss_match, total_step)
    Logger.scalar_summary('loss_job', loss_job, total_step)


