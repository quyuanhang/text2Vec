import torch
from torch import nn, LongTensor
from torch.autograd import Variable
from torch.nn import functional

USE_GPU = torch.cuda.is_available()


class MTL(nn.Module):
    def __init__(self, n_job, n_geek, emb_dim = 36):
        super(MTL, self).__init__()
        self.shape = (n_job, n_geek)
        self.job_emb = nn.Embedding(n_job, emb_dim)
        self.geek_emb = nn.Embedding(n_geek, emb_dim)
        dim = emb_dim * 2
        self.dim = dim
        self.hiden1 = nn.Sequential(
            nn.Linear(dim, dim // 2, bias=False),
            nn.ReLU(),
        )
        self.hiden21 = nn.Sequential(
            nn.Linear(dim // 3, dim // 6, bias=False),
            nn.ReLU(),
            nn.Linear(dim // 6, 1, bias=False),
            nn.Sigmoid()
        )
        self.hiden22 = nn.Sequential(
            nn.Linear(dim // 3, dim // 6, bias=False),
            nn.ReLU(),
            nn.Linear(dim // 6, 1, bias=False),
            nn.Sigmoid()
        )
        self.hiden23 = nn.Sequential(
            nn.Linear(dim // 2, dim // 6, bias=False),
            nn.ReLU(),
            nn.Linear(dim // 6, 1, bias=False),
            nn.Sigmoid()
        )
        self.criterion = nn.MSELoss()

    def forward(self, *input):
        jd, resume, position = input
        jd_emb = self.jd_emb(jd)
        resume_emb = self.resume_emb(resume)
        if self.norm:
            jd_wc = self.word_count(jd)
            resume_wc = self.word_count(resume)
            jd_emb = jd_emb / jd_wc
            resume_emb = resume_emb / resume_wc
        if self.weight:
            position_emb = self.position_emb(position)
            jd_emb = self.weighted(jd_emb, position_emb)
            resume_emb = self.weighted(resume_emb, position_emb)

        jd_emb = torch.sum(jd_emb, dim=1)
        resume_emb = torch.sum(resume_emb, dim=1)

        x = torch.cat((jd_emb, resume_emb), dim=-1).squeeze(dim=1)
        x = self.hiden1(x)
        x1 = self.hiden21(x[:, :self.dim // 3])
        x2 = self.hiden22(x[:, self.dim // 6:])
        x3 = self.hiden23(x)
        x = torch.cat((x1, x2, x3), dim=-1)
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
            position = profile['position'][job_tensor_tmp]
            job_tensor = profile['job'][job_tensor_tmp]
            geeks_tensor = profile['geek'][geeks_tensor]
            if USE_GPU:
                job_tensor = job_tensor.cuda()
                geeks_tensor = geeks_tensor.cuda()
            # print(job_tensor.shape)
            scores = self.forward(job_tensor, geeks_tensor)
            # print(scores)
            scores = scores.cpu()
            scores = scores.detach().numpy()[:, -1]
            predictions.append(scores)
        return predictions

    @staticmethod
    def batch_fit(model, optimizer, sample, profile=False):
        job = sample[:, 0]
        geek = sample[:, 1]
        label = sample[:, 2:]
        job = Variable(LongTensor(job))
        geek = Variable(LongTensor(geek))
        label = label.float()
        position = profile['position'][job]
        job = profile['job'][job]
        geek = profile['geek'][geek]
        label = label.view(label.shape[0], 1)
        if USE_GPU:
            job = job.cuda()
            geek = geek.cuda()
            position = position.cuda()
            label = label.cuda()
        # 前向传播计算损失
        out = model(job, geek, position)
        loss = model.criterion(out, label)
        # 后向传播计算梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item() * label.size(0)

