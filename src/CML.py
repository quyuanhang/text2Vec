import torch
from torch import nn, LongTensor
from torch.autograd import Variable

USE_GPU = torch.cuda.is_available()

class CML(nn.Module):
    def __init__(self, n_job, n_geek, layers_dim):
        super(CML, self).__init__()
        self.job_emb = nn.Embedding(n_job, int(layers_dim))
        self.geek_emb = nn.Embedding(n_geek, int(layers_dim))
        self.criterion = nn.MSELoss()
        self.shape = (n_job, n_geek)

    def forward(self, job, geek):
        # print(job.shape)
        job = self.job_emb(job)
        geek = self.geek_emb(geek)
        # print(job.shape)
        x = job - geek
        # print(x.shape)
        x = torch.norm(x, dim=2)
        return x

    def predict(self, test_data):
        n_job, n_geek = self.shape
        predictions = []
        # with torch.no_grad():
        for sample in test_data:
            job = sample[0]
            if job >= n_job:
                continue
            geeks = [x for x in sample[1:] if x < n_geek]
            job_tensor = Variable(LongTensor([job] * len(geeks))).unsqueeze(dim=1)
            geeks_tensor = Variable(LongTensor(geeks)).unsqueeze(dim=1)
            if USE_GPU:
                job_tensor = job_tensor.cuda()
                geeks_tensor = geeks_tensor.cuda()
            scores = - self.forward(job_tensor, geeks_tensor)
            # print(scores)
            scores = scores.cpu()
            scores = scores.detach().numpy()
            predictions.append(scores)
        return predictions

    @staticmethod
    def batch_fit(model, optimizer, sample):
        job, geek, label = sample.t()
        # job, geek = feature
        job = Variable(LongTensor(job))
        geek = Variable(LongTensor(geek))
        # label = label.view(label.shape[0], 1)
        label = label.float()
        if USE_GPU:
            job = job.cuda()
            geek = geek.cuda()
            label = label.cuda()
        # 前向传播计算损失
        out = model(job, geek)
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
        geekj = Variable(LongTensor(geekj))
        # label = label.view(label.shape[0], 1)
        if USE_GPU:
            job = job.cuda()
            geek = geek.cuda()
            geekj = geekj.cuda()
        # 前向传播计算损失
        out_pos = model(job, geek)
        out_nega = model(job, geekj)
        # hinge_loss = torch.max((0, (delta+out_nega-out_pos)), dim=1)
        # ave_hinge_loss = torch.mean(hinge_loss)
        loss = (out_pos - out_nega)
        loss = torch.mean(loss)
        # 后向传播计算梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item() * sample.size(0)
