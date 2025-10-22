from argparse import ArgumentParser

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--local_rank', default=8, type=int,
                        help='node rank for distributed training')

    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--rot', dest='test_rot', action='store_true')
    parser.add_argument('--half', dest='half', action='store_true')
    parser.add_argument('--pix3d', dest='pix3d', action='store_true')
    parser.add_argument('--abo', dest='abo', action='store_true')
    parser.add_argument('--test_ckpt', type=str,
                        default='pix2vox/output/checkpoints/viat_AVDT/checkpoint_88.pth')

    parser.add_argument('--taxonomy_path', default='pix2vox/datasets/ShapeNetRendering.json')
    parser.add_argument('--pix3d_taxonomy_path', default='/mnt/d/zhouhang/zero123/pix2vox/datasets/Pix3D.json')
    parser.add_argument('--abo_taxonomy_path', default='/mnt/d/zhouhang/zero123/pix2vox/datasets/ABORendering.json')
    parser.add_argument('--trainset_path', default='/mnt/d/zhouhang/zero123/ShapeNetRendering_256RGB/image_align/%s/%s/%02d.png')
    parser.add_argument('--trainset_pix3d_path', default='/mnt/d/zhouhang/zero123/syn_rgb/%s/%s/%02d.png')
    parser.add_argument('--testset_path_all', default='/mnt/d/zhouhang/zero123/ShapeNetRendering_256RGB/image_all/%s/%s/%02d.png')
    parser.add_argument('--testset_path_half', default='/mnt/d/zhouhang/zero123/ShapeNetRendering_256RGB/image_half/%s/%s/%02d.png')
    parser.add_argument('--aug_path', default='/mnt/d/zhouhang/zero123/shapenet_trainset_augmentation/%s/%s/%02d.png')
    parser.add_argument('--voxel_path', default='/mnt/d/zhouhang/zero123/ShapeNetVox32/%s/%s/model.binvox')
    parser.add_argument('--abo_path', default='/mnt/d/zhouhang/zero123/ABO_render/image_rot/%s/%s/%02d.png')
    parser.add_argument('--abo_voxel_path', default='/mnt/d/zhouhang/zero123/ABO_binvox/%s/%s/%s.binvox')
    parser.add_argument('--dist_pool_path', default='/mnt/d/zhouhang/zero123/attack/dist_pool')
    parser.add_argument('--rotation_pool_path', default='/mnt/d/zhouhang/zero123/attack/rotation_pool_%s_degree_%s')
    parser.add_argument('--render_output_path', default='/mnt/d/zhouhang/zero123/attack/AT_%s_degree_%s_')
    parser.add_argument('--mvtn_path', default='/mnt/d/zhouhang/zero123/attack/mvtn_model')

    parser.add_argument('--ckpt_path', type=str,
                        default='/mnt/d/zhouhang/zero123/%s/output/checkpoints/origin_align/checkpoint_best.pth',
                        help='pretrained checkpoint path to load')
    parser.add_argument('--pix3d_ckpt_path', type=str,
                        default='/mnt/d/zhouhang/zero123/%s/output/checkpoints/pix3d/checkpoint-epoch-010.pth',
                        help='pretrained checkpoint path to load')
    parser.add_argument('--model_dir', default='/mnt/d/zhouhang/zero123/%s/output/checkpoints/AT_%s_degree_zero123_105000',
                        help='directory of model for saving checkpoint')
    parser.add_argument('--logs_dir', default='/mnt/d/zhouhang/zero123/%s/output/logs/AT_%s_degree_',
                        help='directory of model for saving checkpoint')

    parser.add_argument('--attack_model', type=str, default='pix2vox')
    parser.add_argument('--interval_degree', type=int, default='30')
    parser.add_argument('--AT_type', type=str, default='ours', choices=['VIAT', 'MVTN', 'random', 'ART', 'ours', 'nature'])
    parser.add_argument('--n_views', type=int, default=3, help='number of views for validate')
    parser.add_argument('--ft_rate', type=float, default=0.001, help='finetune rate')
    parser.add_argument('--num_attack', type=int, default=5, help='number of images for attack')

    parser.add_argument('--gpus', type=str, default=0,
                        help='number of gpus')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--batch_size_render', type=int, default=1,
                        help='images num for mutil gpus')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate')

    parser.add_argument('--iteration', type=int, default=20,
                        help='iteration')
    parser.add_argument('--epochs', type=int, default=100, ## 76,
                        metavar='N',help='number of epochs to train')
    parser.add_argument('--epochs_fintune', type=int, default=1, #10,
                        help='iteration')

    parser.add_argument('--optim_method', type=str, default='NES',
                        choices=['random', 'NES', 'xNES', 'train_pose'],
                        help='num_sample')

    parser.add_argument('--search_num', type=int, default=2,
                        help='search_num')

    parser.add_argument('--popsize', type=int, default=21,
                        help='popsize')

    parser.add_argument('--random_begin', default=False,
                        help='mu random init')

    parser.add_argument('--num_k', type=int, default=5,
                        help='mu random init')

    parser.add_argument('--mu_lamba', type=float, default=0.05,
                        help='iteration')
    parser.add_argument('--sigma_lamba', type=float, default=0.05,
                        help='iteration')
    parser.add_argument('--omiga_lamba', type=float, default=0.05,
                        help='iteration')

    parser.add_argument('--random_eplison', type=float, default=0.01,
                        help='iteration')

    # for AT:

    parser.add_argument('--train_batch_size', type=int, default=8, ##72 * 4, 
                        metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=196, ##72 * 4, 
                        metavar='N', help='input batch size for testing (default: 128)')

    parser.add_argument('--crop_size', type=int, default=224, metavar='N',
                        help='crop_size')
    parser.add_argument('--img_size', type=int, default=256, metavar='N',
                        help='img_size')

    parser.add_argument('--weight-decay', '--wd', default=2e-4,
                        type=float, metavar='W')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--epsilon', default=0.031,
                        help='perturbation')
    parser.add_argument('--num-steps', default=1,
                        help='perturb number of steps')
    parser.add_argument('--step-size', default=0.007,
                        help='perturb step size')
    parser.add_argument('--beta', default=6.0,
                        help='regularization, i.e., 1/lambda in TRADES')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-freq', '-s', default=10, type=int, metavar='N',
                        help='save frequency')

    return parser.parse_args()
