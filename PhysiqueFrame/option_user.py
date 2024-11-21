import argparse


def init():
    parser = argparse.ArgumentParser(description="PyTorch")

    parser.add_argument('--path_to_images', type=str,
                        default='./PAA-3-User',
                        help='directory to images')

    parser.add_argument('--path_to_save_csv', type=str,
                        default="./PhysiqueAA50K_csv/finetune/ISFJ",
                        help='directory to csv_folder')

    parser.add_argument('--experiment_dir_name', type=str, default='./',
                        help='directory to project')
    parser.add_argument('--path_to_model_weight', type=str,
                        default='./PhysiqueFrame_pth/User-ISFJ/PAA-224px_24_7_8_black224_vacc0.7587238285144566_srcc0.6991912269902341vlcc0.7244031047954934_vaccS0.7612163509471586_srccS0.6249889895727923vlccS0.6567651563159452_vaccP0.7637088733798604_srccP0.6635490844480009vlccP0.7030796167964104_0.pth',
                        help='directory to pretrain model')

    # 学习率0.000003， batchsize32时没有出现，图像312*248时没有出现NaN
    parser.add_argument('--init_lr', type=int, default=0.0001, help='learning_rate'
                        )
    parser.add_argument('--num_epoch', type=int, default=10, help='epoch num for train'
                        )
    parser.add_argument('--batch_size', type=int, default=16, help='16how many pictures to process one time'
                        )
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers',
                        )
    # parser.add_argument('--num_workers', type=int, default=2, help ='num_workers',
    #                     )
    # parser.add_argument('--gpu_id', type=str, default='2', help='which gpu to use')
    parser.add_argument('--gpu_id', type=str, default='0', help='which gpu to use')
    parser.add_argument('--llava_gpu_id', type=str, default='3', help='which gpu to use')

    args = parser.parse_args()
    return args
