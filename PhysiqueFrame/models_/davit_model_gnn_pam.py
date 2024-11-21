import torch.nn as nn
import torch
from einops import rearrange
from .pointmlp.pointmlp import PointMLPGenEncoder

import torch.nn.functional as F
from . import clip_vit
from .pam.pam import PoseAnythingModel


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class PhysiqueFrame(nn.Module):
    def __init__(self, model_type='Davit_base'):
        super().__init__()
        self.model_type = 'PAA-224px'

        """ CLIP """
        self.vit = clip_vit.VisionTransformer(224, 14, 1024, 24, 16, 768)
        """ CLIP """

        self.avgpool_point = nn.AdaptiveAvgPool2d(1)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.pointmlp = PointMLPGenEncoder()
      
        self.linear_point_mlp = nn.Linear(654, 256)
        self.x_cat_norm = nn.LayerNorm(256)
        self.linear_body_shape = nn.Sequential(
            # nn.LayerNorm(10),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        self.linear_final = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

        clip_input_num = 768
        mf_input_num = 256
        gt_input_num = 256

        self.batchnorm1d_final = nn.BatchNorm1d(clip_input_num + mf_input_num + gt_input_num)
        self.batchnorm1d_x = nn.BatchNorm1d(clip_input_num)
        self.batchnorm1d_mf = nn.BatchNorm1d(mf_input_num)
        self.batchnorm1d_gt = nn.BatchNorm1d(gt_input_num)

        self.layernorm_x = nn.LayerNorm(clip_input_num)
        self.layernorm_mf = nn.LayerNorm(mf_input_num)
        self.layernorm_gt = nn.LayerNorm(gt_input_num)
        self.groupnorm_mf = nn.GroupNorm(num_channels=mf_input_num, num_groups=16)

        self.classifier4 = nn.Sequential(
            nn.Linear(clip_input_num + mf_input_num + gt_input_num, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.adapter = Adapter(clip_input_num, 4)
        self.ratio = 0.7
        self.kernel_size = 9
        self.pam = PoseAnythingModel()

        self.groupnorm_pam = nn.GroupNorm(num_channels=256, num_groups=16)
        self.layernorm_query = nn.LayerNorm(196)
        people_answer_num = 5

        # mf_input_num + gt_input_num + clip_input_num + people_answer_num
        # mf_input_num + 256 + clip_input_num + people_answer_num
        self.pam_classifier = nn.Sequential(
            nn.Linear(mf_input_num + gt_input_num + clip_input_num , 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.pam_shape_classifier = nn.Sequential(
            nn.Linear(mf_input_num + gt_input_num + clip_input_num , 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.pam_pose_classifier = nn.Sequential(
            nn.Linear(mf_input_num + gt_input_num + clip_input_num , 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.batchnorm1d_pam = nn.LayerNorm(256 + mf_input_num + clip_input_num)

    def forward(self, x, x_clip, mesh_point, body_pose, body_shape, point_position, edge_index, target_s,
                target_weight_s, img_metas, preference_tensor):
        pam_feature, query_feature = self.pam.forward_iaa(
            img_s=x,
            target_s=target_s,
            target_weight_s=target_weight_s,
            target_q=None,
            target_weight_q=None,
            img_metas=img_metas,
        )  # pam_feature: batch*3*15*256
        pam_feature = pam_feature.transpose(1, -1)
        pam_feature = F.adaptive_max_pool2d(pam_feature, 1).squeeze(dim=-1).squeeze(dim=-1)
        pam_feature = self.groupnorm_pam(pam_feature)  # pam_feature batch*256
        #
        mesh_feature = self.pointmlp.forward(mesh_point)  # mesh_feature: batch*256*654  8, 64, 384
        mesh_feature = F.adaptive_max_pool1d(mesh_feature, 1).squeeze(dim=-1)  # mesh_feature: batch*256
        mesh_feature = self.groupnorm_mf(mesh_feature)  # mesh_feature batch*256

        """  CLIP """
        image_features = self.vit.forward(x.type(torch.float32)).float()
        x_clip = self.adapter(image_features)
        image_features = self.ratio * x_clip + (1 - self.ratio) * image_features
        image_features = self.layernorm_x(image_features)
        """  CLIP """

        # x = torch.cat((image_features, pam_feature, mesh_feature, preference_tensor), 1)
        x = torch.cat((image_features, pam_feature, mesh_feature), 1)


        y_pred = self.pam_classifier(x)
        y_shape_pred = self.pam_shape_classifier(x)
        y_pose_pred = self.pam_pose_classifier(x)

        return y_pred * 10, y_shape_pred * 10, y_pose_pred * 10
