import math
import os
import trimesh

import argparse
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from SMPLer_X.common.utils_smpler_x.human_models import smpl_x
# from models_.llava_pp.LLaVA.matcher import LLaVA_Matcher

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
    mean=IMAGE_NET_MEAN,
    std=IMAGE_NET_STD)

import sys
import os.path as osp

sys.path.insert(0, osp.join('./SMPLer_X', 'main'))
sys.path.insert(0, osp.join('./SMPLer_X', 'data'))
sys.path.insert(0, './SMPLer_X')

from SMPLer_X.main.config import cfg
import cv2
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch
import option

opt = option.init()
device = torch.device("cuda:{}".format(opt.gpu_id))

cudnn.benchmark = True



from SMPLer_X.common.base import Demoer

demoer = Demoer()
demoer._make_model(device)
from SMPLer_X.common.utils_smpler_x.preprocessing import load_img, process_bbox, generate_patch_image


demoer.model.eval()
transform = transforms.ToTensor()
sys.path.append('./.cache/torch/hub/ultralytics_yolov5_master')

detector = torch.hub.load('./ultralytics_yolov5_master', 'yolov5s',
                          "./ultralytics_yolov5_master/checkpoints/yolov5s.pt", source='local')


def no_mesh():
    return None, None, None, None, None, None, None


def cal_point_lenght(x, y):
    assert len(x) == len(y)
    index_list = range(0, len(x))
    length = 0.0
    for i in index_list:
        length += (x[i] - y[i]) ** 2
    return math.sqrt(length)


# 计算身高、头长度、身长度等等，坐标都是root-relative 3D coordinates
def cal_length(joint_origin_root_cam):
    # head_point = joint_origin_root_cam[0][15]  # 头的坐标
    # neck_point = joint_origin_root_cam[0][12]  # 脖子的坐标
    # shoulder_point = (joint_origin_root_cam[0][16] + joint_origin_root_cam[0][17]) / 2  # 左右肩膀中间点坐标
    # pelvis_point = joint_origin_root_cam[0][0]  # 骨盆的坐标
    # hip_point = (joint_origin_root_cam[0][1] + joint_origin_root_cam[0][2]) / 2  # 左右臀部中间点坐标
    # hip_left_point = joint_origin_root_cam[0][1]  # 左臀部坐标
    # hip_right_point = joint_origin_root_cam[0][2]  # 右臀部坐标
    # knee_left_point = joint_origin_root_cam[0][3]  # 左膝盖坐标
    # knee_right_point = joint_origin_root_cam[0][4]  # 右膝盖坐标
    # ankle_left_point = joint_origin_root_cam[0][5]  # 左脚踝坐标
    # ankle_right_point = joint_origin_root_cam[0][6]  # 右脚踝坐标
    # heel_left_point = joint_origin_root_cam[0][62]  # 左脚后跟坐标
    # heel_right_point = joint_origin_root_cam[0][65]  # 右脚后跟坐标
    head_point = joint_origin_root_cam[0][24]  # 头的坐标
    neck_point = joint_origin_root_cam[0][7]  # 脖子的坐标
    shoulder_point = (joint_origin_root_cam[0][8] + joint_origin_root_cam[0][9]) / 2  # 左右肩膀中间点坐标
    pelvis_point = joint_origin_root_cam[0][0]  # 骨盆的坐标
    hip_point = (joint_origin_root_cam[0][1] + joint_origin_root_cam[0][2]) / 2  # 左右臀部中间点坐标
    hip_left_point = joint_origin_root_cam[0][1]  # 左臀部坐标
    hip_right_point = joint_origin_root_cam[0][2]  # 右臀部坐标
    knee_left_point = joint_origin_root_cam[0][3]  # 左膝盖坐标
    knee_right_point = joint_origin_root_cam[0][4]  # 右膝盖坐标
    ankle_left_point = joint_origin_root_cam[0][5]  # 左脚踝坐标
    ankle_right_point = joint_origin_root_cam[0][6]  # 右脚踝坐标
    heel_left_point = joint_origin_root_cam[0][16]  # 左脚后跟坐标
    heel_right_point = joint_origin_root_cam[0][19]  # 右脚后跟坐标
    head_neck_length = cal_point_lenght(head_point, neck_point)  # 头长度：头 -> 脖子
    neck_shoulder_length = cal_point_lenght(neck_point, shoulder_point)  # 脖子 -> 左右肩膀
    shoulder_pelvis_length = cal_point_lenght(shoulder_point, pelvis_point)  # 左右肩膀 -> 骨盆
    pelvis_hip_length = cal_point_lenght(pelvis_point, hip_point)  # 骨盆 -> 左右臀部
    # 上半身长度：头 -> 脖子 -> 左右肩膀 -> 骨盆 ->左右臀部
    upper_body_length = head_neck_length + neck_shoulder_length + shoulder_pelvis_length + pelvis_hip_length
    # 左脚长度： 左臀部 -> 左膝盖 -> 左脚踝 -> 左脚后跟
    leg_left_length = cal_point_lenght(hip_left_point, knee_left_point) + cal_point_lenght(knee_left_point,
                                                                                           ankle_left_point) + cal_point_lenght(
        ankle_left_point, heel_left_point)
    # 右脚长度： 右臀部 -> 右膝盖 -> 右脚踝 -> 右脚后跟
    leg_right_length = cal_point_lenght(hip_right_point, knee_right_point) + cal_point_lenght(knee_right_point,
                                                                                              ankle_right_point) + cal_point_lenght(
        ankle_right_point, heel_right_point)
    # 下半身长度：（左脚长度 + 右脚长度）/ 2
    leg_length = (leg_left_length + leg_right_length) / 2
    # 全身长度： 上半身长度 + 下半身长度
    all_body_length = upper_body_length + leg_length

    body_length_param = {}
    body_length_param['upper_body_length'] = upper_body_length  # 上半身长度
    body_length_param['leg_length'] = leg_length  # 下半身长度
    body_length_param['all_body_length'] = all_body_length  # 全身长度
    body_length_param['head_neck_length'] = head_neck_length  # 头的长度

    return body_length_param


def get_body_param(root_cam, body_shape):
    body_length_param = cal_length(root_cam)
    head_body_ratio = body_length_param['all_body_length'] / body_length_param['head_neck_length']
    upper_lower_body_difference = body_length_param['leg_length'] - body_length_param['upper_body_length']
    body_param = {}
    body_param['head_body_ratio'] = head_body_ratio
    body_param['upper_lower_body_difference'] = upper_lower_body_difference
    body_param['all_body_length'] = body_length_param['all_body_length']
    # e = math.exp(1)
    if body_length_param['all_body_length'] == 0:
        body_param['body_beauty_param'] = 0.0
    else:
        # body_param['body_beauty_param'] = - math.log(
        #     -4.431 - 1.367 * math.log(float(body_shape[0]) / (body_length_param['all_body_length'] ** 2.16), e), e)
        body_param['body_beauty_param'] = math.exp(
            -4.431 - 1.367 * math.exp(float(body_shape[0]) / (body_length_param['all_body_length'] ** 2.16)))
    return body_param


def get_mesh(original_img, resize_h):
    # prepare input image
    # original_img = load_img(img_path)
    original_img_height, original_img_width = original_img.shape[:2]

    with torch.no_grad():
        results = detector(original_img)
        person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]
        class_ids, confidences, boxes = [], [], []
        # 判断是否识别到人
        if len(person_results) == 0:
            return no_mesh()
        # choose the main person
        index = 0
        max_box = 0.0
        main_index = index
        for detection in person_results:
            x1, y1, x2, y2, confidence, class_id = detection.tolist()
            if (x2 - x1) * (y2 - y1) > max_box:
                max_box = (x2 - x1) * (y2 - y1)
                main_index = index
            index += 1

        x1, y1, x2, y2, confidence, class_id = person_results[main_index].tolist()
        class_ids.append(class_id)
        confidences.append(confidence)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # for num, indice in enumerate(indices):
        num = 0
        indice = 0
        bbox = boxes[indice]  # x,y,h,w
        bbox = process_bbox(bbox, original_img_width, original_img_height)
        img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False,
                                                               cfg.input_img_shape)
        inputs = transform(img.astype(np.float32)) / 255
        inputs = inputs.cuda()[None, :, :, :]
        inputs = {'img': inputs}
        targets = {}
        meta_info = {}
        # mesh recovery
        out = demoer.model(inputs, targets, meta_info, 'test')
        # 判断是否生成mesh
        if len(out['smplx_mesh_cam']) == 0:
            return no_mesh()

        # mesh_point = -out['smplx_mesh_cam'][0]
        mesh_point = -out['smplx_mesh_cam_zero_pose'][0]
        mesh = trimesh.Trimesh(vertices=mesh_point.to('cpu'), faces=smpl_x.face)
        volume = -mesh.volume

        body_pose = torch.cat([out['smplx_jaw_pose'], out['smplx_root_pose'], out['smplx_body_pose']], 1)
        body_shape = out['smplx_shape']

        joint_proj = out['smplx_joint_proj'].detach().cpu().numpy()[0]
        joint_proj[:, 0] = joint_proj[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
        joint_proj[:, 1] = joint_proj[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
        joint_proj = np.concatenate((joint_proj, np.ones_like(joint_proj[:, :1])), 1)
        joint_proj = np.dot(bb2img_trans, joint_proj.transpose(1, 0)).transpose(1, 0)
        # joint_proj[:, 0] = joint_proj[:, 0] / original_img_width * resize_h
        # joint_proj[:, 1] = joint_proj[:, 1] / original_img_height * resize_h
        # joint_proj[np.where(joint_proj < 0)] = 0
        # joint_proj[np.where(joint_proj >= resize_h)] = resize_h - 1

    return mesh_point, body_pose, body_shape, out['joint_origin_root_cam'], [x1, y1, x2, y2], joint_proj[
                                                                                              0:25], volume


def get_graph_point_position(joint_proj):
    # 返回坐标点 point_position =（x，y）
    point_position = joint_proj[0:14]
    append_position = [int((joint_proj[22][0] + joint_proj[23][0]) / 2),
                       int((joint_proj[22][1] + joint_proj[23][1]) / 2)]
    append_position = np.array(append_position)
    append_position = np.resize(append_position, [1, 2])
    point_position = np.append(point_position, append_position, axis=0).astype(np.int32)
    return point_position


from models_.pam.pipelines import TopDownGenerateTargetFewShot


class AVADataset(Dataset):
    def __init__(self, path_to_csv, images_path, if_train):

        # self.matcher = LLaVA_Matcher(model_path="models_/llava_pp/LLaVA/LLaVA-Phi-3-mini-4k-instruct")
        # self.matcher.init_model()
        self.genHeatMap = TopDownGenerateTargetFewShot()
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.if_train = if_train
        self.h = 224
        if if_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalize
            ])
            self.transform_clip = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((self.h, self.h)),
                transforms.ToTensor(),
                # normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # normalize
            ])
            self.transform_clip = transforms.Compose([
                transforms.Resize((self.h, self.h)),
                transforms.ToTensor(),
                # normalize
            ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        # 分数
        score = np.array([row[f'score']])
        score_shape = np.array([row[f'score_shape']])
        score_pose = np.array([row[f'score_pose']])
        # 照片
        pic_name = row[f'pic_name']
        pic_path = os.path.join(self.images_path, pic_name)

        pic = default_loader(pic_path)
        x = self.transform(pic)
        h = x.shape[1]
        w = x.shape[2]
        if h > w:
            x = transforms.Resize((self.h, int((self.h * w) / h)))(x)
        else:
            x = transforms.Resize((int((self.h * h) / w), self.h))(x)
        x = transforms.CenterCrop(self.h)(x)
        mesh_x = np.transpose(x.numpy(), (1, 2, 0)) * 255
        x = normalize(x)
        # 生成Mesh
        mesh_point, body_pose, body_shape, root_cam, boxes, joint_proj_resize, volume = get_mesh(
            mesh_x, self.h)
        # mesh_point, body_pose, body_shape, root_cam, boxes, joint_proj_resize, volume = self.mesh_point_dict[pic_name]

        if mesh_point != None:
            boxes = torch.tensor(boxes) / 1.0
            mesh_point = torch.tensor(mesh_point) / 1.0
            body_pose = torch.tensor(body_pose) / 1.0
            body_shape = torch.tensor(body_shape) / 1.0

            mesh_point = mesh_point.to("cpu")
            body_pose = body_pose.to("cpu")
            body_shape = body_shape.to("cpu")
            boxes = boxes.to("cpu")
            point_position = get_graph_point_position(joint_proj_resize)


            body_param = get_body_param(root_cam=root_cam, body_shape=body_shape[0])
            body_param_tensor = torch.tensor([body_param['head_body_ratio'], body_param['upper_lower_body_difference'],
                                              body_param['all_body_length'], body_param['body_beauty_param']])

            body_shape = torch.cat(
                (body_shape[:, 0].reshape(1, 1), body_param_tensor.reshape(1, body_param_tensor.shape[0])), 1)
            import torch.nn as nn
            body_shape = (body_shape - torch.mean(body_shape, dim=1, keepdim=True)) / torch.std(body_shape, dim=1,
                                                                                                keepdim=True)

        else:
            mesh_point = torch.zeros(10475, 3) / 1.0
            body_pose = torch.zeros(1, 69) / 1.0
            body_shape = torch.zeros(1, 5) / 1.0
            boxes = torch.zeros(1, 4) / 1.0
            point_position = torch.zeros(15, 2).numpy()
            bmi = 23.0
        edge_index = [
            # 下半身
            [0, 1], [0, 2], [1, 3], [3, 5], [2, 4], [4, 6],
            [1, 0], [2, 0], [3, 1], [5, 3], [4, 2], [6, 4],
            # 上半身不包括头
            [0, 8], [0, 9], [8, 10], [10, 12], [9, 11], [11, 13],
            [8, 0], [9, 0], [10, 8], [12, 10], [11, 9], [13, 11],
            # 其他
            [8, 9], [0, 7], [7, 14], [7, 8], [7, 9],
            [0, 8], [7, 0], [14, 7], [8, 7], [9, 7]
        ]
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_index = edge_index.to("cpu")
        channel_cfg = dict(
            num_output_channels=1,
            dataset_joints=1,
            dataset_channel=[[0, ], ],
            inference_channel=[0, ],
            max_kpt_num=100)
        data_cfg = dict(
            image_size=np.array([self.h, self.h]),
            heatmap_size=[64, 64],
            num_output_channels=channel_cfg['num_output_channels'],
            num_joints=channel_cfg['dataset_joints'],
            dataset_channel=channel_cfg['dataset_channel'],
            inference_channel=channel_cfg['inference_channel'])
        data_cfg['joint_weights'] = None
        data_cfg['use_different_joint_weights'] = False
        kp_src = torch.tensor(point_position).float()

        kp_src_3d = torch.cat((kp_src, torch.zeros(kp_src.shape[0], 1)), dim=-1)
        kp_src_3d_weight = torch.cat((torch.ones_like(kp_src), torch.zeros(kp_src.shape[0], 1)), dim=-1)
        target_s, target_weight_s = self.genHeatMap._msra_generate_target(data_cfg, kp_src_3d, kp_src_3d_weight,
                                                                          sigma=2)
        target_s = torch.tensor(target_s).float()[None]
        target_weight_s = torch.tensor(target_weight_s).float()[None]
        img_metas = [{'sample_skeleton': [edge_index],
                      'query_skeleton': edge_index,
                      'sample_joints_3d': [kp_src_3d],
                      'query_joints_3d': kp_src_3d,
                      'sample_center': [kp_src.mean(dim=0)],
                      'query_center': kp_src.mean(dim=0),
                      'sample_scale': [kp_src.max(dim=0)[0] - kp_src.min(dim=0)[0]],
                      'query_scale': kp_src.max(dim=0)[0] - kp_src.min(dim=0)[0],
                      'sample_rotation': [0],
                      'query_rotation': 0,
                      'sample_bbox_score': [1],
                      'query_bbox_score': 1,
                      'query_image_file': '',
                      'sample_image_file': [''],
                      }]
        target_s = target_s.to("cpu")
        target_weight_s = target_weight_s.to("cpu")
        x_clip = self.transform_clip(pic)

        preference_tensor = eval(row[f'preference_score_list'])
        preference_tensor = torch.tensor(preference_tensor).to("cpu")

        return x, x_clip, score.astype('float32'), score_shape.astype('float32'), score_pose.astype(
            'float32'), mesh_point, body_pose, body_shape, point_position, edge_index, target_s, target_weight_s, img_metas, preference_tensor
