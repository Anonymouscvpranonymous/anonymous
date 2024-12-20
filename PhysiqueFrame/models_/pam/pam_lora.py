import numpy as np
import torch
import cv2

from pam_mmpose.models import builder
from pam_mmpose.models.builder import POSENETS
from pam_mmpose.models.detectors.base import BasePose

from .backbone.swin_utils import load_pretrained
from .backbone.swin_transformer_v2 import SwinTransformerV2
from .keypoint_heads.head import PoseHead


class PoseAnythingModel(BasePose):
    """Few-shot keypoint detectors.
    Args:
        keypoint_head (dict): Keypoint head to process feature.
        encoder_config (dict): Config for encoder. Default: None.
        pretrained (str): Path to the pretrained models.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
    """

    def __init__(self):
        super().__init__()
        self.encoder_config = dict(
            type='SwinTransformerV2',
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=14,
            pretrained_window_sizes=[12, 12, 12, 6],
            drop_path_rate=0.1,
            img_size=224,
        )
        self.keypoint_head_input = dict(
            type='PoseHead',
            in_channels=1024,
            transformer=dict(
                type='EncoderDecoderLora',
                d_model=256,
                nhead=8,
                num_encoder_layers=3,
                num_decoder_layers=3,
                graph_decoder='pre',
                dim_feedforward=1024,
                dropout=0.1,
                similarity_proj_dim=256,
                dynamic_proj_dim=128,
                activation="relu",
                normalize_before=False,
                return_intermediate_dec=True),
            share_kpt_branch=False,
            num_decoder_layer=3,
            with_heatmap_loss=True,

            heatmap_loss_weight=2.0,
            support_order_dropout=-1,
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True))
        self.pretrained = "./models_/pam/pretrained/swinv2_base_22k_500k.pth"
        self.encoder_sample, self.backbone_type = self.init_backbone(self.pretrained, self.encoder_config)
        self.keypoint_head = builder.build_head(self.keypoint_head_input)
        self.keypoint_head.init_weights()
        self.train_cfg = dict(),
        self.test_cfg = dict(
            flip_test=False,
            post_process='default',
            shift_heatmap=True,
            modulate_kernel=11)
        self.target_type = self.test_cfg.get('target_type',
                                             'GaussianHeatMap')  # GaussianHeatMap

    def init_backbone(self, pretrained, encoder_config):
        if 'swin' in pretrained:
            encoder_sample = builder.build_backbone(encoder_config)
            pretext_model = torch.load(pretrained, map_location='cpu')['model']
            model_dict = encoder_sample.state_dict()
            state_dict = {(k.replace('encoder.', '') if k.startswith('encoder.') else k.replace('decoder.', '')): v for
                          k, v in pretext_model.items() if
                          (k.replace('encoder.', '') if k.startswith('encoder.') else k.replace('decoder.',
                                                                                                '')) in model_dict.keys()
                          and model_dict[
                              (k.replace('encoder.', '') if k.startswith('encoder.') else k.replace('decoder.',
                                                                                                    ''))].shape ==
                          pretext_model[k].shape}
            model_dict.update(state_dict)
            encoder_sample.load_state_dict(model_dict)
            # load_pretrained(pretrained, encoder_sample, logger=None)
            backbone = 'swin'
        return encoder_sample, backbone

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.encoder_sample.init_weights(pretrained)
        self.encoder_query.init_weights(pretrained)
        self.keypoint_head.init_weights()

    def forward(self,
                img_s,
                img_q,
                target_s=None,
                target_weight_s=None,
                target_q=None,
                target_weight_q=None,
                img_metas=None,
                return_loss=True,
                **kwargs):
        """Defines the computation performed at every call."""

        if return_loss:
            return self.forward_train(img_s, target_s, target_weight_s, img_q,
                                      target_q, target_weight_q, img_metas,
                                      **kwargs)
        else:
            return self.forward_test(img_s, target_s, target_weight_s, img_q,
                                     target_q, target_weight_q, img_metas,
                                     **kwargs)

    def forward_dummy(self, img_s, target_s, target_weight_s, img_q, target_q,
                      target_weight_q, img_metas, **kwargs):
        return self.predict(
            img_s, target_s, target_weight_s, img_q, img_metas)

    def forward_train(self,
                      img_s,
                      target_s,
                      target_weight_s,
                      img_q,
                      target_q,
                      target_weight_q,
                      img_metas,
                      **kwargs):

        """Defines the computation performed at every call when training."""
        bs, _, h, w = img_q.shape

        output, initial_proposals, similarity_map, mask_s = self.predict(
            img_s, target_s, target_weight_s, img_q, img_metas)

        # parse the img meta to get the target keypoints
        target_keypoints = self.parse_keypoints_from_img_meta(img_metas, output.device, keyword='query')
        target_sizes = torch.tensor([img_q.shape[-2], img_q.shape[-1]]).unsqueeze(0).repeat(img_q.shape[0], 1, 1)

        # if return loss
        losses = dict()
        if self.with_keypoint:
            keypoint_losses = self.keypoint_head.get_loss(
                output, initial_proposals, similarity_map, target_keypoints,
                target_q, target_weight_q * mask_s, target_sizes)
            losses.update(keypoint_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(output[-1],
                                                                target_keypoints,
                                                                target_weight_q * mask_s,
                                                                target_sizes,
                                                                height=h)
            losses.update(keypoint_accuracy)

        return losses

    def forward_test(self,
                     img_s,
                     target_s,
                     target_weight_s,
                     img_q,
                     target_q,
                     target_weight_q,
                     img_metas=None,
                     **kwargs):

        """Defines the computation performed at every call when testing."""
        batch_size, _, img_height, img_width = img_q.shape

        output, initial_proposals, similarity_map, _ = self.predict(img_s, target_s, target_weight_s, img_q, img_metas)
        predicted_pose = output[-1].detach().cpu().numpy()  # [bs, num_query, 2]

        result = {}
        if self.with_keypoint:
            keypoint_result = self.keypoint_head.decode(img_metas, predicted_pose, img_size=[img_width, img_height])
            result.update(keypoint_result)

        result.update({
            "points":
                torch.cat((initial_proposals, output.squeeze())).cpu().numpy()
        })
        result.update({"sample_image_file": img_metas[0]['sample_image_file']})

        return result

    def forward_iaa(self,
                    img_s,
                    target_s,
                    target_weight_s,
                    img_q,
                    target_q,
                    target_weight_q,
                    img_metas=None,
                    **kwargs):

        """Defines the computation performed at every call when testing."""

        batch_size, _, img_height, img_width = img_q.shape
        assert [i['sample_skeleton'][0] != i['query_skeleton'] for i in img_metas]
        skeleton = [i['sample_skeleton'][0] for i in img_metas]

        feature_q, feature_s = self.extract_features(img_s, img_q)

        mask_s = target_weight_s[0]
        for index in range(target_weight_s.shape[1]):
            mask_s = mask_s * target_weight_s[:, index]

        outs_dec, initial_proposals, similarity_map, query_embed = self.keypoint_head.forward_iaa(feature_q, feature_s,
                                                                                                  target_s,
                                                                                                  mask_s, skeleton)

        return outs_dec.transpose(1, 0), query_embed

    def predict(self,
                img_s,
                target_s,
                target_weight_s,
                img_q,
                img_metas=None):

        batch_size, _, img_height, img_width = img_q.shape
        assert [i['sample_skeleton'][0] != i['query_skeleton'] for i in img_metas]
        skeleton = [i['sample_skeleton'][0] for i in img_metas]

        feature_q, feature_s = self.extract_features(img_s, img_q)

        mask_s = target_weight_s[0]
        for target_weight in target_weight_s:
            mask_s = mask_s * target_weight

        output, initial_proposals, similarity_map = self.keypoint_head(feature_q, feature_s, target_s, mask_s, skeleton)

        return output, initial_proposals, similarity_map, mask_s

    def extract_features(self, img_s, img_q):
        if self.backbone_type == 'swin':
            feature_q = self.encoder_sample.forward_features(img_q)  # [bs, C, h, w]
            feature_s = torch.clone(feature_q)
        elif self.backbone_type == 'dino':
            batch_size, _, img_height, img_width = img_q.shape
            feature_q = self.encoder_sample.get_intermediate_layers(img_q, n=1)[0][:, 1:] \
                .reshape(batch_size, img_height // 8, img_width // 8, -1).permute(0, 3, 1, 2)  # [bs, 3, h, w]
            feature_s = [self.encoder_sample.get_intermediate_layers(img, n=1)[0][:, 1:].
                             reshape(batch_size, img_height // 8, img_width // 8, -1).permute(0, 3, 1, 2) for img in
                         img_s]
        elif self.backbone_type == 'dinov2':
            batch_size, _, img_height, img_width = img_q.shape
            feature_q = self.encoder_sample.get_intermediate_layers(img_q, n=1, reshape=True)[0]  # [bs, c, h, w]
            feature_s = [self.encoder_sample.get_intermediate_layers(img, n=1, reshape=True)[0] for img in img_s]
        else:
            feature_s = [self.encoder_sample(img) for img in img_s]
            feature_q = self.encoder_query(img_q)

        return feature_q, feature_s

    def parse_keypoints_from_img_meta(self, img_meta, device, keyword='query'):
        """Parse keypoints from the img_meta.

        Args:
            img_meta (dict): Image meta info.
            device (torch.device): Device of the output keypoints.
            keyword (str): 'query' or 'sample'. Default: 'query'.

        Returns:
            Tensor: Keypoints coordinates of query images.
        """

        if keyword == 'query':
            query_kpt = torch.stack([
                torch.tensor(info[f'{keyword}_joints_3d']).to(device)
                for info in img_meta
            ], dim=0)[:, :, :2]  # [bs, num_query, 2]
        else:
            query_kpt = []
            for info in img_meta:
                if isinstance(info[f'{keyword}_joints_3d'][0], torch.Tensor):
                    samples = torch.stack(info[f'{keyword}_joints_3d'])
                else:
                    samples = np.array(info[f'{keyword}_joints_3d'])
                query_kpt.append(torch.tensor(samples).to(device)[:, :, :2])
            query_kpt = torch.stack(query_kpt, dim=0)  # [bs, , num_samples, num_query, 2]
        return query_kpt

    # UNMODIFIED
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_limb_color=None,
                    radius=4,
                    text_color=(255, 0, 0),
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_limb_color (np.array[Mx3]): Color of M limbs.
                If None, do not draw limbs.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """

        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape

        bbox_result = []
        pose_result = []
        for res in result:
            bbox_result.append(res['bbox'])
            pose_result.append(res['keypoints'])

        if len(bbox_result) > 0:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            mmcv.imshow_bboxes(
                img,
                bboxes,
                colors=bbox_color,
                top_k=-1,
                thickness=thickness,
                show=False,
                win_name=win_name,
                wait_time=wait_time,
                out_file=None)

            for person_id, kpts in enumerate(pose_result):
                # draw each point on image
                if pose_kpt_color is not None:
                    assert len(pose_kpt_color) == len(kpts), (
                        len(pose_kpt_color), len(kpts))
                    for kid, kpt in enumerate(kpts):
                        x_coord, y_coord, kpt_score = int(kpt[0]), int(
                            kpt[1]), kpt[2]
                        if kpt_score > kpt_score_thr:
                            img_copy = img.copy()
                            r, g, b = pose_kpt_color[kid]
                            cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                       radius, (int(r), int(g), int(b)), -1)
                            transparency = max(0, min(1, kpt_score))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)

                # draw limbs
                if skeleton is not None and pose_limb_color is not None:
                    assert len(pose_limb_color) == len(skeleton)
                    for sk_id, sk in enumerate(skeleton):
                        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1,
                                                                  1]))
                        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1,
                                                                  1]))
                        if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                                and pos1[1] < img_h and pos2[0] > 0
                                and pos2[0] < img_w and pos2[1] > 0
                                and pos2[1] < img_h
                                and kpts[sk[0] - 1, 2] > kpt_score_thr
                                and kpts[sk[1] - 1, 2] > kpt_score_thr):
                            img_copy = img.copy()
                            X = (pos1[0], pos2[0])
                            Y = (pos1[1], pos2[1])
                            mX = np.mean(X)
                            mY = np.mean(Y)
                            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                            angle = math.degrees(
                                math.atan2(Y[0] - Y[1], X[0] - X[1]))
                            stickwidth = 2
                            polygon = cv2.ellipse2Poly(
                                (int(mX), int(mY)),
                                (int(length / 2), int(stickwidth)), int(angle),
                                0, 360, 1)

                            r, g, b = pose_limb_color[sk_id]
                            cv2.fillConvexPoly(img_copy, polygon,
                                               (int(r), int(g), int(b)))
                            transparency = max(
                                0,
                                min(
                                    1, 0.5 *
                                       (kpts[sk[0] - 1, 2] + kpts[sk[1] - 1, 2])))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)

        show, wait_time = 1, 1

        return img
