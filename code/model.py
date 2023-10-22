import torch
import torch.nn as nn
import utils


def weight_SVDHead(src, src_corr, weight):
    weight = weight.unsqueeze(1)
    src2 = (src * weight).sum(dim = 2, keepdim = True) / weight.sum(dim = 2, keepdim = True)
    src_corr2 = (src_corr * weight).sum(dim = 2, keepdim = True)/weight.sum(dim = 2,keepdim = True)
    src_centered = src - src2
    src_corr_centered = src_corr - src_corr2
    H = torch.matmul(src_centered * weight, src_corr_centered.transpose(2, 1).contiguous())

    R = []

    for i in range(src.size(0)):
        u, s, v = torch.svd(H[i])
        r = torch.matmul(v, u.transpose(1, 0)).contiguous()
        r_det = torch.det(r)

        if r_det<0:
            u, s, v = torch.svd(H[i])
            reflect = nn.Parameter(torch.eye(3), requires_grad=False).cuda()
            reflect[2, 2] = -1
            v = torch.matmul(v, reflect)
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
        R.append(r)

    R = torch.stack(R, dim = 0).cuda()

    t = torch.matmul(-R, src2.mean(dim = 2, keepdim=True)) + src_corr2.mean(dim = 2, keepdim = True)
    return R, t.view(src.size(0), 3)

def SVDHead(src, src_corr):
    src_centered = src - src.mean(dim=2, keepdim=True)
    src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

    H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

    R = []

    for i in range(src.size(0)):
        u, s, v = torch.svd(H[i])
        r = torch.matmul(v, u.transpose(1, 0).contiguous())
        r_det = torch.det(r)
        if r_det < 0:
            u, s, v = torch.svd(H[i])
            reflect = nn.Parameter(torch.eye(3), requires_grad=False).cuda()
            reflect[2, 2] = -1
            v = torch.matmul(v, reflect)
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
        R.append(r)

    R = torch.stack(R, dim=0)

    t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
    return R, t.view(src.size(0), 3)

def mean(data):
    return torch.sum(data)/(data.size(0)*data.size(1)*data.size(2))


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20):
    idx = knn(x, k=k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature


class HTMC(nn.Module):
    def __init__(self, args):
        super(HTMC, self).__init__()


        self.en_block1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.en_block3 = nn.Sequential(nn.Conv2d(64, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.en_block5 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, bias=False), nn.BatchNorm2d(512), nn.ReLU())
        self.encode = nn.Sequential(self.en_block1, self.en_block3, self.en_block5)

        self.global_feat1 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False), nn.BatchNorm1d(256), nn.ReLU())
        self.global_feat2 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False), nn.BatchNorm1d(128), nn.ReLU())
        self.global_feat3 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False), nn.BatchNorm1d(64), nn.ReLU())
        self.global_feat = nn.Sequential(self.global_feat1, self.global_feat2, self.global_feat3)

        self.de_block5 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.de_block3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.de_block1 = nn.Sequential(nn.Conv2d(64, 2, kernel_size=1, bias=False))
        self.decode = nn.Sequential(self.de_block5, self.de_block3, self.de_block1, nn.Softmax(dim=1))

        self.mask1 = nn.Sequential(nn.Conv1d(512+64+64, 512, kernel_size=1, bias=False), nn.BatchNorm1d(512), nn.ReLU())
        self.mask2 = nn.Sequential(nn.Conv1d(512, 128, kernel_size=1, bias=False), nn.BatchNorm1d(128), nn.ReLU())
        self.mask3 = nn.Sequential(nn.Conv1d(128, 2, kernel_size=1, bias=False))

        self.mask = nn.Sequential(self.mask1, self.mask2, self.mask3, nn.Softmax(dim=1))



    def bulid_graph(self, feature, idx):
        batch_size, num_points, k = idx.size()
        device = torch.device('cuda')
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

        num_dims = feature.size(1)

        feature = feature.transpose(2, 1).contiguous()
        feature = feature.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        feature = feature.permute(0, 3, 1, 2)
        return feature

    def Mask(self, probability):

        threshold = 0.8
        while True:
            mask = torch.where(probability > threshold, 1.0, 0)

            if torch.sum(mask) > 100:
                break
            threshold *= 0.9
            if threshold < 0.00001:
                return torch.ones_like(probability)
        return mask

    def one_to_bool(self, mat):
        return mat > 0.5

    def Registration_one(self, src, tgt, correspondence_relation):
        correspondence_mask = self.one_to_bool(self.Mask(correspondence_relation)).view(-1)
        src_size, tgt_szie = src.size(1), tgt.size(1)
        src = src.unsqueeze(dim=2).repeat(1, 1, tgt_szie).view(src.size(0), -1).transpose(0, 1)[correspondence_mask].transpose(0, 1)
        src_corr = tgt.unsqueeze(dim=2).repeat(1, 1, src_size).transpose(1, 2).contiguous().view(tgt.size(0), -1).transpose(0, 1)[correspondence_mask].transpose(0, 1)


        src, src_corr = src.unsqueeze(0), src_corr.unsqueeze(0)


        src, src_corr = utils.global_spacial_consistency(src, src_corr)

        src, src_corr = src.squeeze(0), src_corr.squeeze(0)

        src_centered = src - src.mean(dim=1, keepdim=True)
        src_corr_centered = src_corr - src_corr.mean(dim=1, keepdim=True)

        H = torch.matmul(src_centered, src_corr_centered.transpose(1, 0).contiguous())

        u, s, v = torch.svd(H)
        r = torch.matmul(v, u.transpose(1, 0).contiguous())
        r_det = torch.det(r)
        if r_det < 0:
            u, s, v = torch.svd(H)
            reflect = torch.eye(3)
            reflect[2, 2] = -1
            reflect = reflect.cuda()
            v = torch.matmul(v, reflect)
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
        t = torch.matmul(-r, src.mean(dim=1, keepdim=True)) + src_corr.mean(dim=1, keepdim=True)
        t = t.squeeze(1)

        return r, t

    def registration(self, src, tgt, correspondence_relation):
        R = []
        T = []
        for i in range(src.size(0)):
            r, t = self.Registration_one(src[i], tgt[i], correspondence_relation[i])
            R.append(r)
            T.append(t)

        R = torch.stack(R, dim=0)
        T = torch.stack(T, dim=0)
        return R, T

    def Registration_evaluate(self, src, tgt, r, t):
        transformed_src = torch.matmul(r, src) + t.unsqueeze(1)
        dis = utils.dis_torch(transformed_src, tgt)
        min_dis = torch.min(dis, dim=1)[0]
        num_inlier = torch.sum(min_dis < utils.dense_torch(src.unsqueeze(0))/2)
        return num_inlier

    def SmartRegistration(self, src, tgt, correspondence_pre, src_mask_pre, tgt_mask_pre):
        R = []
        T = []
        correspondence_masked = []
        for i in range(src.size(0)):
            correspondence_relation = correspondence_pre[i]

            index = 1

            src_mask_base = (torch.abs(src_mask_pre[i] - 0.5)/(src_mask_pre[i] - 0.5)) * torch.abs(src_mask_pre[i] - 0.5)**index / (0.5**index * 2) + 1
            tgt_mask_base = (torch.abs(tgt_mask_pre[i] - 0.5)/(tgt_mask_pre[i] - 0.5)) * torch.abs(tgt_mask_pre[i] - 0.5)**index / (0.5**index * 2) + 1

            src_mask_ex = src_mask_base.unsqueeze(1)
            tgt_mask_ex = tgt_mask_base.unsqueeze(0)
            correspondence_relation_ba = correspondence_relation * src_mask_ex * tgt_mask_ex
            try:
                r_ba, t_ba = self.Registration_one(src[i], tgt[i], correspondence_relation_ba)
                num_inlier_balance = self.Registration_evaluate(src[i], tgt[i], r_ba, t_ba)
            except:
                num_inlier_balance = 0

            src_mask_ex = torch.where(src_mask_pre[i] > 0.5, src_mask_base, 1).unsqueeze(1)
            tgt_mask_ex = torch.where(tgt_mask_pre[i] > 0.5, tgt_mask_base, 1).unsqueeze(0)
            correspondence_relation_re = correspondence_relation * src_mask_ex * tgt_mask_ex
            try:
                r_re, t_re = self.Registration_one(src[i], tgt[i], correspondence_relation_re)
                num_inlier_recall = self.Registration_evaluate(src[i], tgt[i], r_re, t_re)
            except:
                num_inlier_recall = 0
            src_mask_ex = torch.where(src_mask_pre[i] > 0.5, 1, src_mask_base).unsqueeze(1)
            tgt_mask_ex = torch.where(tgt_mask_pre[i] > 0.5, 1, tgt_mask_base).unsqueeze(0)
            correspondence_relation_pr = correspondence_relation * src_mask_ex * tgt_mask_ex
            try:
                r_pr, t_pr = self.Registration_one(src[i], tgt[i], correspondence_relation_pr)
                num_inlier_precision = self.Registration_evaluate(src[i], tgt[i], r_pr, t_pr)
            except:
                num_inlier_precision = 0


            try:
                r_or, t_or = self.Registration_one(src[i], tgt[i], correspondence_relation)
                num_inlier_origin = self.Registration_evaluate(src[i], tgt[i], r_or, t_or)
            except:
                num_inlier_origin = 0


            found = False
            if max(num_inlier_balance, num_inlier_recall, num_inlier_precision, num_inlier_origin) == num_inlier_balance:
                if not found:
                    found = True
                    R.append(r_ba)
                    T.append(t_ba)
                    correspondence_masked.append(correspondence_relation_ba)

            if max(num_inlier_balance, num_inlier_recall, num_inlier_precision, num_inlier_origin) == num_inlier_recall:
                if not found:
                    found = True
                    R.append(r_re)
                    T.append(t_re)
                    correspondence_masked.append(correspondence_relation_re)

            if max(num_inlier_balance, num_inlier_recall, num_inlier_precision, num_inlier_origin) == num_inlier_precision:
                if not found:
                    found = True
                    R.append(r_pr)
                    T.append(t_pr)
                    correspondence_masked.append(correspondence_relation_pr)

            if max(num_inlier_balance, num_inlier_recall, num_inlier_precision, num_inlier_origin) == num_inlier_origin:
                if not found:
                    R.append(r_or)
                    T.append(t_or)
                    correspondence_masked.append(correspondence_relation)

        R = torch.stack(R, dim=0)
        T = torch.stack(T, dim=0)
        correspondence_masked = torch.stack(correspondence_masked)
        return R, T, correspondence_masked

    def forward(self, src, tgt):
        src_feat = get_graph_feature(src)
        tgt_feat = get_graph_feature(tgt)

        src_feat = self.encode(src_feat)
        src_feat = src_feat.max(dim=-1)[0]
        src_global = self.global_feat(src_feat).max(dim=-1)[0].unsqueeze(-1).repeat(1, 1, 768)

        tgt_feat = self.encode(tgt_feat)
        tgt_feat = tgt_feat.max(dim=-1)[0]
        tgt_global = self.global_feat(tgt_feat).max(dim=-1)[0].unsqueeze(-1).repeat(1, 1, 768)

        src_mask_feat = torch.cat((src_feat, src_global, tgt_global), dim=1)
        tgt_mask_feat = torch.cat((tgt_feat, tgt_global, src_global), dim=1)
        src_mask = self.mask(src_mask_feat)
        tgt_mask = self.mask(tgt_mask_feat)


        src_feat_expand = src_feat.unsqueeze(dim=-1).repeat(1, 1, 1, tgt.size(2))
        tgt_feat_expand = tgt_feat.unsqueeze(dim=-1).repeat(1, 1, 1, src.size(2)).transpose(2, 3)
        similarity = torch.mul(src_feat_expand, tgt_feat_expand)
        correspondence_relation = self.decode(similarity)[:, 0, :, :]


        return correspondence_relation, src_mask[:,0,:], tgt_mask[:,0,:]
