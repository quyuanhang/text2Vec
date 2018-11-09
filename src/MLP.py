import torch
from torch import nn, LongTensor, FloatTensor
from torch.autograd import Variable

USE_GPU = torch.cuda.is_available()

class MLP(nn.Module):
    def __init__(self, n_word, n_positin, shape, emb_dim, with_bias, pretrain_emb=None):
        super(MLP, self).__init__()
        self.shape = shape
        self.jd_emb = nn.Embedding(n_word, emb_dim)
        self.resume_emb = nn.Embedding(n_word, emb_dim)
        if len(pretrain_emb) > 0:
            self.jd_emb.weight.data = FloatTensor(pretrain_emb.copy())
            self.resume_emb.weight.data = FloatTensor(pretrain_emb.copy())

        self.job_emb = nn.Embedding(n_positin, emb_dim)
        dim = emb_dim * 2
        self.hidens = nn.Sequential(
            nn.Linear(dim, dim // 2, bias=with_bias),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4, bias=with_bias),
            nn.ReLU(),
            nn.Linear(dim // 4, 1, bias=with_bias),
            nn.Sigmoid()
        )
        # self.cross = nn.Bilinear(emb_dim, emb_dim, emb_dim)
        self.criterion = nn.BCELoss()

    def forward(self, *input):
        jd, resume, jobid = input
        # print(jd.shape)
        jd = self.jd_emb(jd)
        jd = torch.sum(jd, dim=1)
        jd = jd / torch.norm(jd, p=2, dim=-1).unsqueeze(dim=1)
        resume = self.resume_emb(resume)
        resume = torch.sum(resume, dim=1)
        resume = resume / torch.norm(resume, p=2, dim=-1).unsqueeze(dim=1)
        # x = self.hidens(jd, resume)
        # x = self.cross(jd, resume)
        # print(jd.shape)
        x = torch.cat((jd, resume), dim=-1)
        # print(x.shape)
        x = self.hidens(x)
        # print(x.shape)
        # x = x.squeeze(dim=1)
        return x

    def predict(self, test_data, profile):
        n_job, n_geek = self.shape
        predictions = []
        # with torch.no_grad():
        for sample in test_data:
            job = sample[0]
            if job >= n_job:
                continue
            geeks = [x for x in sample[1:] if x < n_geek]
            job_tensor_tmp = Variable(LongTensor([job] * len(geeks)))
            geeks_tensor = Variable(LongTensor(geeks))
            # job_tensor = job_tensor.view(job_tensor.shape[0], 1)
            # geeks_tensor = geeks_tensor.view(geeks_tensor.shape[0], 1)
            if profile:
                position = profile['position'][job_tensor_tmp]
                job_tensor = profile['job'][job_tensor_tmp]
                geeks_tensor = profile['geek'][geeks_tensor]
            if USE_GPU:
                job_tensor = job_tensor.cuda()
                geeks_tensor = geeks_tensor.cuda()
                position = position.cuda()
            # print(job_tensor.shape)
            scores = self.forward(job_tensor, geeks_tensor, position)
            # print(scores)
            scores = scores.cpu()
            scores = scores.detach().numpy()
            predictions.append(scores)
        return predictions

    @staticmethod
    def batch_fit(model, optimizer, sample, profile=False):
        # job, geek, label = sample.t()
        job = sample[:, 0].unsqueeze(dim=1)
        geek = sample[:, 1].unsqueeze(dim=1)
        label = sample[:, 2].unsqueeze(dim=1)
        # print('job size \n', job.shape)
        job_tmp = Variable(LongTensor(job))
        geek = Variable(LongTensor(geek))
        # label = label.view(label.shape[0], 1)
        label = label.float()
        if profile:
            position = profile['position'][job_tmp]
            job = profile['job'][job_tmp].squeeze()
            geek = profile['geek'][geek].squeeze()
        if USE_GPU:
            job = job.cuda()
            geek = geek.cuda()
            label = label.cuda()
        # 前向传播计算损失
        out = model(job, geek, position)
        loss = model.criterion(out, label)
        # 后向传播计算梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item() * label.size(0)

    @staticmethod
    def batch_fit_pairwise(model, optimizer, sample, delta=0):
        # job, geek, label = sample.t()
        job = sample[:, 0].unsqueeze(dim=1)
        geek = sample[:, 1].unsqueeze(dim=1)
        geekj = sample[:, 2].unsqueeze(dim=1)
        # print('job size \n', job.shape)
        job = Variable(LongTensor(job))
        geek = Variable(LongTensor(geek))
        # label = label.view(label.shape[0], 1)
        if USE_GPU:
            job = job.cuda()
            geek = geek.cuda()
            geekj = geekj.cuda()
        # 前向传播计算损失
        out_pos = model(job, geek)
        out_nega = model(job, geekj)
        # hinge_loss = torch.nn.functional.relu(out_nega-out_pos)
        # loss = torch.mean(hinge_loss)
        loss = torch.nn.functional.logsigmoid(out_nega - out_pos)
        loss = torch.mean(loss)
        # 后向传播计算梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item() * sample.size(0)

