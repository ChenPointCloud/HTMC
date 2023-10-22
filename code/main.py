import torch
import datasets
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import model
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import utils
from tqdm import tqdm
import torch.nn.functional as F


mask = True
hie = True

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()

log = None
regis_recall = None

features_grad = 0
def extract(g):
    global features_grad
    features_grad = g


def save_arr(path, data):
    with open(path, "w") as file:
        for num in data:
            file.write(str(num) + " ")

def save_PC(path, data):
    data = data[0].cpu().numpy().T
    np.savetxt(path, data)


def correspondence_mask_calculate(src, tgt, r, t, dense_factor=0.5):
    transformed_src = torch.matmul(r, src) + t.unsqueeze(2)
    distance_map = utils.pairwise_distance_batch(transformed_src, tgt)
    correspondence_mask = distance_map < utils.dense_torch(tgt) * dense_factor
    correspondence_mask = torch.where(correspondence_mask, 1.0, 0.0)
    return correspondence_mask


def hierarchical_mask(src, tgt, R, t):
    transformed_src = torch.matmul(R, src) + t.unsqueeze(2)
    distance_map = utils.pairwise_distance_batch(transformed_src, tgt)
    if hie:
        correspondence_mask = distance_map > utils.dense_torch(tgt) * 1.5
        correspondence_mask1 = torch.where(correspondence_mask, 0.1, 0.5)
        correspondence_mask = distance_map > utils.dense_torch(tgt)
        correspondence_mask2 = torch.where(correspondence_mask, 0.0, 0.3)
        correspondence_mask = distance_map > utils.dense_torch(tgt) * 0.5
        correspondence_mask3 = torch.where(correspondence_mask, 0.0, 0.1)
        mask = correspondence_mask1 + correspondence_mask2 + correspondence_mask3
    else:
        correspondence_mask = distance_map > utils.dense_torch(tgt) * 0.5
        mask = torch.where(correspondence_mask, 0.0, 1)
    return mask

def mask_evaluate(src, tgt, r, t, mask_pre, dense_factor=0.5):
    mask_gt = correspondence_mask_calculate(src, tgt, r, t, dense_factor)

    ac = torch.mean(torch.eq(mask_gt, mask_pre).float())
    recall = torch.sum(torch.logical_and(mask_gt, mask_pre).float()) / torch.sum(mask_gt)
    precision = torch.sum(torch.logical_and(mask_gt, mask_pre).float()) / torch.sum(mask_pre)

    return ac, recall, precision

def train_one_epoch(args, sim, train_loader, opt_sim):
    correspondence_losses = 0
    mask_losses = 0

    acs = []
    precisions = []
    recalls = []
    acs_mask = []
    precisions_mask = []
    recalls_mask = []

    for datas in tqdm(train_loader):
        src = datas[0].cuda()
        tgt = datas[1].cuda()
        R = datas[2].cuda()
        t = datas[3].cuda()
        src_mask_gt = datas[4].float().cuda()
        tgt_mask_gt = datas[5].float().cuda()



        sim_outputs = sim(src, tgt)
        correspondence_pre = sim_outputs[0]
        src_mask_pre, tgt_mask_pre = sim_outputs[1], sim_outputs[2]


        correspondence_mask = correspondence_mask_calculate(src, tgt, R, t)
        hie_mask = hierarchical_mask(src, tgt, R, t)


        if mask:
            index = 1
            src_mask_ex = ((src_mask_pre - 0.5)**index / (0.5**index * 2) + 1).unsqueeze(2)
            tgt_mask_ex = ((tgt_mask_pre - 0.5)**index / (0.5**index * 2) + 1).unsqueeze(1)
            correspondence_pre_mask = correspondence_pre * src_mask_ex.detach() * tgt_mask_ex.detach()
        else:
            correspondence_pre_mask = correspondence_pre

        bceLoss = torch.nn.BCELoss()
        bce_non = torch.nn.BCELoss(reduction='none')
        if hie:
            positive_mask = torch.where(correspondence_mask > 0.9, 1.0, 0.0)
            positive_loss = torch.sum((F.relu(hie_mask - correspondence_pre)**2) * positive_mask)
            negative_mask = torch.where(correspondence_mask < 0.9, 0.1, 0.0)
            negative_loss = torch.sum((F.relu(correspondence_pre - hie_mask)**2) * negative_mask)
            correspondence_loss = positive_loss + negative_loss
        else:
            weight = torch.where(correspondence_mask > 0.9, 1.0, 0.1)
            correspondence_loss = torch.sum(bce_non(correspondence_pre, hie_mask) * weight)

        mask_loss = bceLoss(src_mask_pre, src_mask_gt) + bceLoss(tgt_mask_pre, tgt_mask_gt)

        if mask:
            loss = correspondence_loss + mask_loss
        else:
            loss = correspondence_loss
        correspondence_losses += correspondence_loss.item()
        mask_losses += mask_loss.item()
        correspondence_mask_pre = sim.Mask(correspondence_pre)

        ac = 1 - torch.mean((correspondence_mask-correspondence_mask_pre)**2)
        recall = 1 - torch.sum(torch.nn.functional.relu(correspondence_mask-correspondence_mask_pre))/torch.sum(correspondence_mask)
        precision = 1 - torch.sum(torch.nn.functional.relu(correspondence_mask_pre-correspondence_mask))/torch.sum(correspondence_mask_pre)
        acs.append(ac.detach().cpu())
        recalls.append(recall.detach().cpu())
        precisions.append(precision.detach().cpu())



        correspondence_pre_mask_zo = sim.Mask(correspondence_pre_mask)
        ac_mask = 1 - torch.mean((correspondence_mask-correspondence_pre_mask_zo)**2)
        recall_mask = 1 - torch.sum(torch.nn.functional.relu(correspondence_mask-correspondence_pre_mask_zo))/torch.sum(correspondence_mask)
        precision_mask = 1 - torch.sum(torch.nn.functional.relu(correspondence_pre_mask_zo-correspondence_mask))/torch.sum(correspondence_pre_mask_zo)
        acs_mask.append(ac_mask.detach().cpu())
        recalls_mask.append(recall_mask.detach().cpu())
        precisions_mask.append(precision_mask.detach().cpu())


        loss.backward()
        opt_sim.step()
        opt_sim.zero_grad()


    acs = sum(acs) / len(acs)
    recalls = sum(recalls) / len(recalls)
    precisions = sum(precisions) / len(precisions)
    acs_mask = sum(acs_mask) / len(acs_mask)
    recalls_mask = sum(recalls_mask) / len(recalls_mask)
    precisions_mask = sum(precisions_mask) / len(precisions_mask)
    log.cprint('accuracy:%s' % acs)
    log.cprint('recall:%s' % recalls)
    log.cprint('precisions:%s' % precisions)

    log.cprint('accuracy_mask:%s' % acs_mask)
    log.cprint('recall_mask:%s' % recalls_mask)
    log.cprint('precisions_mask:%s' % precisions_mask)

    log.cprint("correspondence loss:%f, mask_loss:%f" % (correspondence_losses/len(train_loader), mask_losses/len(train_loader)))

    return recalls.item(), precisions.item(), recalls_mask.item(), precisions_mask.item(), mask_losses/len(train_loader)

def test_one_epoch(args, sim, test_loader, registration=False):
    rotations = []
    translations = []
    rotations_pre = []
    translations_pre = []

    acs = []
    precisions = []
    recalls = []
    acs_mask = []
    precisions_mask = []
    recalls_mask = []

    flag = 0
    fra = open('debug/flag.txt', 'w')
    for datas in tqdm(test_loader):
        src = datas[0].cuda()
        tgt = datas[1].cuda()
        R = datas[2].cuda()
        t = datas[3].cuda()

        sim_outputs = sim(src, tgt)
        correspondence_pre = sim_outputs[0]
        src_mask_pre, tgt_mask_pre = sim_outputs[1], sim_outputs[2]


        if registration:
            if mask:
                R_pre, t_pre, correspondence_pre_mask = sim.SmartRegistration(src, tgt, correspondence_pre, src_mask_pre, tgt_mask_pre)
            else:
                correspondence_pre_mask = correspondence_pre
                R_pre, t_pre = sim.registration(src, tgt, correspondence_pre)
        else:
            if mask:
                src_mask_ex = (src_mask_pre * 2).unsqueeze(2)
                tgt_mask_ex = (tgt_mask_pre * 2).unsqueeze(1)
                correspondence_pre_mask = correspondence_pre * src_mask_ex * tgt_mask_ex
            else:
                correspondence_pre_mask = correspondence_pre


        correspondence_mask_pre = sim.Mask(correspondence_pre)
        ac, recall, precision = mask_evaluate(src, tgt, R, t, correspondence_mask_pre)
        acs.append(ac.detach().cpu())
        recalls.append(recall.detach().cpu())
        precisions.append(precision.detach().cpu())

        correspondence_pre_mask_zo = sim.Mask(correspondence_pre_mask)
        ac_mask, recall_mask, precision_mask = mask_evaluate(src, tgt, R, t, correspondence_pre_mask_zo)
        acs_mask.append(ac_mask.detach().cpu())
        recalls_mask.append(recall_mask.detach().cpu())
        precisions_mask.append(precision_mask.detach().cpu())

        euler = utils.npmat2euler(R.detach().cpu())
        euler = euler[0:1, :]
        euler_pre = utils.npmat2euler(R_pre.detach().cpu())
        euler_pre = euler_pre[0:1, :]

        r_rmse = np.sqrt(np.mean((euler - euler_pre) ** 2))
        if r_rmse > 5:
            save_cloud(tgt, 'debug/tgt' + str(flag) + '.txt')
            transformed_src = torch.matmul(R, src) + t.unsqueeze(2)
            save_cloud(transformed_src, 'debug/transformed_src' + str(flag) + '.txt')
            transformed_src_pre = torch.matmul(R_pre, src) + t_pre.unsqueeze(2)
            save_cloud(transformed_src_pre, 'debug/transformed_src_pre' + str(flag) + '.txt')
            flag += 1
            fra.write(str(flag) + ':' + str(r_rmse) + '\n')
            if flag > 15:
                break
            print(r_rmse)

        if registration:
            rotations.append(R.detach().cpu())
            translations.append(t.detach().cpu())
            rotations_pre.append(R_pre.detach().cpu())
            translations_pre.append(t_pre.detach().cpu())

    if registration:
        rotations = np.concatenate(rotations, axis=0)
        translations = np.concatenate(translations, axis=0)
        rotations_pre = np.concatenate(rotations_pre, axis=0)
        translations_pre = np.concatenate(translations_pre, axis=0)
        euler = utils.npmat2euler(rotations)
        euler_pre = utils.npmat2euler(rotations_pre)
        r_rmse = np.sqrt(np.mean((euler - euler_pre) ** 2))
        r_mae = np.mean(np.abs(euler - euler_pre))
        t_rmse = np.sqrt(np.mean((translations - translations_pre) ** 2))
        t_mae = np.mean(np.abs(translations - translations_pre))


    numpy_recalls_mask = [tensor.cpu().numpy() for tensor in recalls_mask]
    numpy_precisions_mask = [tensor.cpu().numpy() for tensor in precisions_mask]

    np.savetxt('recalls_mask_batch.txt', numpy_recalls_mask)
    np.savetxt('precisions_mask_batch.txt', numpy_precisions_mask)

    acs = sum(acs) / len(acs)
    recalls = sum(recalls) / len(recalls)
    precisions = sum(precisions) / len(precisions)
    acs_mask = sum(acs_mask) / len(acs_mask)
    recalls_mask = sum(recalls_mask) / len(recalls_mask)
    precisions_mask = sum(precisions_mask) / len(precisions_mask)

    log.cprint('accuracy%s' % acs)
    log.cprint('recall%s' % recalls)
    log.cprint('precisions%s' % precisions)

    log.cprint('accuracy_mask:%s' % acs_mask)
    log.cprint('recall_mask:%s' % recalls_mask)
    log.cprint('precisions_mask:%s' % precisions_mask)


    if registration:
        regis_recall = utils.registration_recall(euler, euler_pre, args.inlier_r, translations, translations_pre, args.inlier_t)
        log.cprint('regis_recall %f' % regis_recall)
        log.cprint("test: rot_rmse:%f rot_mae：%f trans_rmse:%f trans_mae:%f" % (r_rmse, r_mae, t_rmse, t_mae))
        return r_rmse, r_mae, t_rmse, t_mae


def train(args, train_loader, test_loader):
    sim = model.HTMC(args).cuda()
    opt_sim = optim.Adam(sim.parameters(), lr=args.lr, weight_decay=1e-4)


    scheduler_sim = MultiStepLR(opt_sim, milestones=[15, 30], gamma=0.1)

    recall_all = []
    precision_all = []
    recall_mask_all = []
    precision_mask_all = []
    mask_loss_all = []

    for i in range(args.epochs):
        log.cprint("Epoch:%d" % i)
        recalls, precisions, recalls_mask, precisions_mask, mask_loss = train_one_epoch(args, sim, train_loader, opt_sim)
        torch.save(sim.state_dict(), 'checkpoints/%s/models/sim.%d.t7' % (args.exp_name, i))

        recall_all.append(recalls)
        precision_all.append(precisions)
        recall_mask_all.append(recalls_mask)
        precision_mask_all.append(precisions_mask)
        mask_loss_all.append(mask_loss)

        with torch.no_grad():
            test_one_epoch(args, sim, test_loader)

        scheduler_sim.step()


    save_arr('checkpoints/%s/recall.txt'% (args.exp_name), recall_all)
    save_arr('checkpoints/%s/precision.txt'% (args.exp_name), precision_all)
    save_arr('checkpoints/%s/recall_mask.txt'% (args.exp_name), recall_mask_all)
    save_arr('checkpoints/%s/precision_mask.txt'% (args.exp_name), precision_mask_all)
    save_arr('checkpoints/%s/mask_loss.txt'% (args.exp_name), mask_loss_all)


def test(args, test_loader, epoch=49):
    sim = model.HTMC(args).cuda()
    sim.load_state_dict(torch.load('checkpoints/%s/models/sim.%s.t7' % (args.exp_name, epoch)), strict=True)
    print("Epoch:%s" % epoch)

    return test_one_epoch(args, sim, test_loader, True)

def save_cloud(cloud, name):
    tmp = cloud[0]
    tmp = tmp.to('cpu').detach()
    tmp = tmp.numpy().T
    np.savetxt(name, tmp)

def main(args):

    if args.dataset == 'modelnet40':
        train_loader = DataLoader(
            datasets.ModelNet40(num_points=1024, num_subsampled_points=768, partition='train', gaussian_noise=args.gaussian,
                       unseen=args.unseen, rot_factor=args.factor),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            datasets.ModelNet40(num_points=1024, num_subsampled_points=768, partition='test', gaussian_noise=args.gaussian,
                       unseen=args.unseen, rot_factor=args.factor),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == 'human':
        train_loader = DataLoader(
            datasets.Human(args, partition='train'),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            datasets.Human(args, partition='test'),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == 'icl':
        train_loader = DataLoader(
            datasets.ICL(args, partition='train'),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            datasets.ICL(args, partition='test'),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == 'kitti':
        train_loader = DataLoader(
            datasets.KITTI(args, partition='train'),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            datasets.KITTI(args, partition='test'),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)


    global log
    log = IOStream('checkpoints/' + args.exp_name + '/run.log')

    if args.eval is False:
        train(args, train_loader, test_loader)
        with torch.no_grad():
            test(args, test_loader, '34')
    else:
        with torch.no_grad():
            test(args, test_loader, '34')

def get_args():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=2, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=2, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=35, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--gaussian', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--sub_points', type=int, default=768, metavar='N',
                        help='partial overlapping')
    parser.add_argument('--dataset', type=str, default='human', metavar='N')
    parser.add_argument('--max_inlier_ratio', type=int, default=80, metavar='N')
    parser.add_argument('--min_inlier_ratio', type=int, default=30, metavar='N')
    parser.add_argument('--inlier_r', type=float, default=1, metavar='M')
    parser.add_argument('--inlier_t', type=float, default=0.1, metavar='M')
    parser.add_argument('--least_inlier_points', type=int, default=100, metavar='M')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N')

    args = parser.parse_args()
    return args

def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')

def load_txt(filename):
    a = np.loadtxt(filename)
    a = np.random.permutation(a)[:768, :].astype('float32')
    return a

if __name__ == '__main__':

    args = get_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    _init_(args)
    main(args)

