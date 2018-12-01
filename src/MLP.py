import torch
from torch import nn, LongTensor, FloatTensor
from torch.autograd import Variable

USE_GPU = torch.cuda.is_available()

class MLP(nn.Module):
    def __init__(self, n_word, n_positin, shape, emb_dim, with_bias, pretrain_emb=[], weight=False, norm=False):
        super(MLP, self).__init__()
        self.shape = shape
        self.jd_emb = nn.Embedding(n_word, emb_dim)
        self.resume_emb = nn.Embedding(n_word, emb_dim)
        if len(pretrain_emb) != 0:
            self.jd_emb.weight.data = FloatTensor(pretrain_emb.copy())
            self.resume_emb.weight.data = FloatTensor(pretrain_emb.copy())
            self.jd_emb.weight.requires_grad = False
            self.resume_emb.weight.requires_grad = False

        self.weight = weight
        self.norm = norm
        self.position_emb = nn.Embedding(n_positin, emb_dim)
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

    def weighted(self, jd_emb, position_emd):
        position_t = position_emd.permute(0, 2, 1)
        jd_weight = torch.bmm(jd_emb, position_t)
        jd_weight = torch.nn.functional.softmax(jd_weight, dim=1)
        jd_emb = jd_emb * jd_weight
        return jd_emb

    def normalize(self, jd_emb):
        jd_emb = torch.sum(jd_emb, dim=1)
        jd_emb = jd_emb / torch.norm(jd_emb, p=2, dim=-1).unsqueeze(dim=1)
        return jd_emb

    def word_count(self, jd):
        c = torch.sum(jd!=0, dim=1)
        c = c.view(c.shape[0], 1, 1)
        c = c.float()
        return c

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

        # jd_emb = torch.sum(jd_emb, dim=1)
        # resume_emb = torch.sum(resume_emb, dim=1)
        jd_emb = torch.max(jd_emb, dim=1)[0]
        resume_emb = torch.max(resume_emb, dim=1)[0]
        x = torch.cat((jd_emb, resume_emb), dim=-1)
        x = self.hidens(x)
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
        job, geek, label = sample.t()
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

if __name__ == '__main__':
    model = MLP(20, 10, (5, 5), 3, False, weight=True)
    jd = LongTensor([
        [1, 2, 3],
        [4, 5, 6]
    ])
    resume = LongTensor([
        [3, 2],
        [5, 6]
    ])
    job = LongTensor([
        [1],
        [2]
    ])
    model.forward(jd, resume, job)
    # for row in model.named_parameters():
    #     print(row)