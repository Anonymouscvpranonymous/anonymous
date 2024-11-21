import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# from torch import nn
from tqdm import tqdm
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score

from models_.davit_model_gnn_pam import PhysiqueFrame

from dataset_user import AVADataset
from util import EDMLoss, AverageMeter
import option_user as option
import nni
from nni.utils import merge_parameter

opt = option.init()
device = torch.device("cuda:{}".format(opt.gpu_id))


def adjust_learning_rate(params, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = params['init_lr'] * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_score(opt, y_pred):
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor)
    w = w.to(device)

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np


def create_data_part(opt):
    train_csv_path = os.path.join(opt['path_to_save_csv'], 'train.csv')
    val_csv_path = os.path.join(opt['path_to_save_csv'], 'val.csv')
    test_csv_path = os.path.join(opt['path_to_save_csv'], 'test.csv')

    train_ds = AVADataset(train_csv_path, opt['path_to_images'], if_train=True)
    val_ds = AVADataset(val_csv_path, opt['path_to_images'], if_train=False)
    test_ds = AVADataset(test_csv_path, opt['path_to_images'], if_train=False)

    train_loader = DataLoader(train_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)

    return train_loader, val_loader, test_loader


def train(opt, model, loader, optimizer, criterion, writer=None, global_step=None, name=None):
    model.train()
    train_losses = AverageMeter()
    for idx, (
            x, x_clip, y, y_shape, y_pose, mesh_point, body_pose, body_shape, point_position, edge_index, target_s,
            target_weight_s,
            img_metas, preference_tensor) in enumerate(tqdm(loader)):
        x = x.to(device)
        x_clip = x_clip.to(device)
        y = y.type(torch.FloatTensor)
        y = y.to(device)
        y_shape = y_shape.to(device)
        y_pose = y_pose.to(device)
        mesh_point = mesh_point.to(device)
        body_pose = body_pose.to(device)
        body_shape = body_shape.to(device)
        point_position = point_position.to(device)
        edge_index = edge_index.to(device)
        target_s = target_s.to(device)
        target_weight_s = target_weight_s.to(device)
        preference_tensor = preference_tensor.to(device)

        y_pred, y_shape_pred, y_pose_pred = model(x, x_clip, mesh_point, body_pose, body_shape, point_position,
                                                  edge_index, target_s,
                                                  target_weight_s,
                                                  img_metas, preference_tensor)

        loss = criterion(y, y_pred, y_shape, y_shape_pred, y_pose, y_pose_pred).requires_grad_(True)

        optimizer.zero_grad()

        loss.backward()

        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()

        optimizer.step()
        train_losses.update(loss.item(), x.size(0))

        if writer is not None:
            writer.add_scalar(f"{name}/train_loss.avg", train_losses.avg, global_step=global_step + idx)

    return train_losses.avg


def validate(opt, model, loader, criterion, writer=None, global_step=None, name=None, test_or_valid_flag='test'):
    model.eval()
    validate_losses = AverageMeter()
    true_score = []
    pred_score = []
    true_score_shape = []
    true_score_pose = []
    pred_score_shape = []
    pred_score_pose = []

    with torch.no_grad():
        for idx, (
                x, x_clip, y, y_shape, y_pose, mesh_point, body_pose, body_shape, point_position, edge_index, target_s,
                target_weight_s, img_metas, preference_tensor) in enumerate(tqdm(loader)):

            x = x.to(device)
            x_clip = x_clip.to(device)
            y = y.type(torch.FloatTensor)
            y = y.to(device)
            y_shape = y_shape.to(device)
            y_pose = y_pose.to(device)
            mesh_point = mesh_point.to(device)
            body_pose = body_pose.to(device)
            body_shape = body_shape.to(device)
            point_position = point_position.to(device)
            edge_index = edge_index.to(device)
            target_s = target_s.to(device)
            target_weight_s = target_weight_s.to(device)
            preference_tensor = preference_tensor.to(device)

            y_pred, y_shape_pred, y_pose_pred = model(x, x_clip, mesh_point, body_pose, body_shape, point_position,
                                                      edge_index, target_s,
                                                      target_weight_s,
                                                      img_metas, preference_tensor)

            for i in y.data.cpu().numpy().tolist():
                true_score.append(i)
            for j in y_pred.data.cpu().numpy().tolist():
                pred_score.append(j)
            for i in y_shape.data.cpu().numpy().tolist():
                true_score_shape.append(i)
            for j in y_shape_pred.data.cpu().numpy().tolist():
                pred_score_shape.append(j)
            for i in y_pose.data.cpu().numpy().tolist():
                true_score_pose.append(i)
            for j in y_pose_pred.data.cpu().numpy().tolist():
                pred_score_pose.append(j)

            # 对分数预测的误差
            # loss = (criterion(y, y_pred) + criterion(y_shape, y_shape_pred) + criterion(y_pose,y_pose_pred)).requires_grad_(True)
            loss = criterion(y, y_pred, y_shape, y_shape_pred, y_pose, y_pose_pred).requires_grad_(True)

            validate_losses.update(loss.item(), x.size(0))

            if writer is not None:
                writer.add_scalar(f"{name}/val_loss.avg", validate_losses.avg, global_step=global_step + idx)
    # 皮尔森和斯皮尔曼 这个会和Apex形成NaNs，在训练时可以取消，在验证时重启
    # lcc_mean = [0,0]
    # srcc_mean = [0,0]
    lcc_mean = pearsonr(np.array(pred_score).ravel(), np.array(true_score).ravel())
    srcc_mean = spearmanr(pred_score, true_score)
    lcc_mean_shape = pearsonr(np.array(pred_score_shape).ravel(), np.array(true_score_shape).ravel())
    srcc_mean_shape = spearmanr(pred_score_shape, true_score_shape)
    lcc_mean_pose = pearsonr(np.array(pred_score_pose).ravel(), np.array(true_score_pose).ravel())
    srcc_mean_pose = spearmanr(pred_score_pose, true_score_pose)

    true_score = np.array(true_score)
    # 分数小于5，则是0，bad图像，若是1，则是好图像
    true_score_lable = np.where(true_score <= 5, 0, 1)
    pred_score = np.array(pred_score)
    pred_score_lable = np.where(pred_score <= 5, 0, 1)
    # 最后是判断二分类的准确性
    acc = accuracy_score(true_score_lable, pred_score_lable)

    true_score_shape = np.array(true_score_shape)
    # 分数小于5，则是0，bad图像，若是1，则是好图像
    true_score_lable_shape = np.where(true_score_shape <= 5, 0, 1)
    pred_score_shape = np.array(pred_score_shape)
    pred_score_lable_shape = np.where(pred_score_shape <= 5, 0, 1)
    # 最后是判断二分类的准确性
    acc_shape = accuracy_score(true_score_lable_shape, pred_score_lable_shape)

    true_score_pose = np.array(true_score_pose)
    # 分数小于5，则是0，bad图像，若是1，则是好图像
    true_score_lable_pose = np.where(true_score_pose <= 5, 0, 1)
    pred_score_pose = np.array(pred_score_pose)
    pred_score_lable_pose = np.where(pred_score_pose <= 5, 0, 1)
    # 最后是判断二分类的准确性
    acc_pose = accuracy_score(true_score_lable_pose, pred_score_lable_pose)
    # print("acc:{0}".format(acc))
    # print("validate_losses:{0}".format(validate_losses.avg))
    # 对分数预测的误差、对二分类的精度、皮尔森、斯皮尔曼
    print('{}, accuracy: {}, lcc_mean: {}, srcc_mean: {}, validate_losses: {}'.format(test_or_valid_flag, acc,
                                                                                      lcc_mean[0], srcc_mean[0],
                                                                                      validate_losses.avg))
    return validate_losses.avg, acc, acc_shape, acc_pose, lcc_mean, srcc_mean, lcc_mean_shape, srcc_mean_shape, lcc_mean_pose, srcc_mean_pose


def start_train(opt):
    train_loader, val_loader, test_loader = create_data_part(opt)

    model = PhysiqueFrame()
    load_pth_path = opt['path_to_model_weight']

    model_load = torch.load(load_pth_path,
                            map_location='cuda:' + opt['gpu_id'])
    model_dict = model.state_dict()

    state_dict = {}
    for k, v in model_load.items():
        if k in model_dict.keys() and model_dict[k].shape == model_load[k].shape:
            state_dict[k] = v

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    for name, param in model.named_parameters():
        if name.startswith('vit'):
            param.requires_grad = False

    import util
    criterion = util.CustomMultiLoss()
    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), eps=1e-05)

    writer = SummaryWriter(log_dir=os.path.join(opt['experiment_dir_name'], 'logs'))
    srcc_best = 0.0
    vacc_best = 0.0
    for e in range(opt['num_epoch']):
        adjust_learning_rate(opt, optimizer, e)
        train_loss = train(opt, model=model, loader=train_loader, optimizer=optimizer, criterion=criterion,
                           writer=writer, global_step=len(train_loader) * e,
                           name=f"{opt['experiment_dir_name']}_by_batch")
        val_loss, vacc, acc_shape, acc_pose, vlcc, vsrcc, lcc_mean_shape, srcc_mean_shape, lcc_mean_pose, srcc_mean_pose = validate(
            opt,
            model=model,
            loader=val_loader,
            criterion=criterion,
            writer=writer,
            global_step=len(
                val_loader) * e,
            name=f"{opt['experiment_dir_name']}_by_batch",
            test_or_valid_flag='valid')
        # test_loss, tacc, tlcc, tsrcc = validate(opt, model=model, loader=test_loader, criterion=criterion,
        #                                         writer=writer, global_step=len(val_loader) * e,
        #                                         name=f"{opt['experiment_dir_name']}_by_batch",
        #                                         test_or_valid_flag='test')

        if ((vsrcc[0] > srcc_best or vacc > vacc_best)) and ((vacc > 0.95) and vsrcc[0] > 0.69):
            srcc_best = vsrcc[0]
            vacc_best = vacc
            model_savetime = model.model_type + '_24_7_8' + '_black224'
            model_name = f"{model_savetime}_vacc{vacc}_srcc{vsrcc[0]}vlcc{vlcc[0]}_vaccS{acc_shape}_srccS{srcc_mean_shape[0]}vlccS{lcc_mean_shape[0]}_vaccP{acc_pose}_srccP{srcc_mean_pose[0]}vlccP{lcc_mean_pose[0]}_{e}.pth"
            torch.save(model.state_dict(), os.path.join(opt['experiment_dir_name'], model_name))

        nni.report_intermediate_result(
            {'default': vacc, 'acc_shape': acc_shape, 'acc_pose': acc_pose, "train_loss": train_loss, "vsrcc": vsrcc[0],
             "vlcc": vlcc[0], "vsrcc_shape": srcc_mean_shape[0], "vlcc_shape": lcc_mean_shape[0],
             "vsrcc_pose": srcc_mean_pose[0], "vlcc_pose": lcc_mean_pose[0], "val_loss": val_loss})
        # nni.report_intermediate_result({'default': vacc, "test_acc": tacc, "val_srcc": tsrcc, "val_lcc": tlcc})
    nni.report_final_result(
        {'default': vacc, 'acc_shape': acc_shape, 'acc_pose': acc_pose, "train_loss": train_loss, "vsrcc": vsrcc[0],
         "vlcc": vlcc[0], "vsrcc_shape": srcc_mean_shape[0], "vlcc_shape": lcc_mean_shape[0],
         "vsrcc_pose": srcc_mean_pose[0], "vlcc_pose": lcc_mean_pose[0], "val_loss": val_loss})
    writer.close()
    # f.close()


# def start_chec
def get_score_one_image():
    pass


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')
    print(os.getcwd())
    #### train model
    # start_train(opt)
    #### test model  用的是"./pretrain_model/u_model.pth"
    # start_check_model(opt)

    # [96, 92, 88, 84, 80, 76, 72, 68, 64, 60, 56, 52]

    # get parameters form tuner   使用NNI调参工具，但是要修改opt给参形式为opt['init_lr']
    tuner_params = nni.get_next_parameter()
    # logger.debug(tuner_params)
    params = vars(merge_parameter(opt, tuner_params))
    print(params)
    start_train(params)
