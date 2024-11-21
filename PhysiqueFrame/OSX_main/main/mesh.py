import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.append("..")
# print(sys.path)
from .config import cfg
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default="1")
    parser.add_argument('--img_path', type=str, default='input5.jpg')
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--encoder_setting', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--decoder_setting', type=str, default='normal',
                        choices=['normal', 'wo_face_decoder', 'wo_decoder'])
    parser.add_argument('--pretrained_model_path', type=str, default='./osx_l.pth.tar')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# load model
cfg.set_additional_args(encoder_setting=args.encoder_setting, decoder_setting=args.decoder_setting,
                        pretrained_model_path=args.pretrained_model_path)
from common.base import Demoer

demoer = Demoer()
demoer._make_model()
from common.utils.preprocessing import load_img, process_bbox, generate_patch_image

model_path = args.pretrained_model_path
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))

demoer.model.eval()
transform = transforms.ToTensor()
sys.path.append('./.cache/torch/hub/ultralytics_yolov5_master')

detector = torch.hub.load('./.cache/torch/hub/ultralytics_yolov5_master', 'yolov5s',
                          "./.cache/torch/hub/checkpoints/yolov5s.pt", source='local')


def no_mesh():
    return None, None, None


def get_mesh(img_path):
    # prepare input image
    original_img = load_img(img_path)
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
        for num, indice in enumerate(indices):
            bbox = boxes[indice]  # x,y,h,w
            bbox = process_bbox(bbox, original_img_width, original_img_height)
            img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
            img = transform(img.astype(np.float32)) / 255
            img = img.cuda()[None, :, :, :]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

        # mesh recovery
        with torch.no_grad():
            out = demoer.model(inputs, targets, meta_info, 'test')
        # 判断是否生成mesh
        if len(out['smplx_mesh_cam']) == 0:
            return no_mesh()

        mesh_point = -out['smplx_mesh_cam'][0]
        body_pose = out['smplx_body_pose']
        body_shape = out['smplx_shape']

    return mesh_point, body_pose, body_shape
