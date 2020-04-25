# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import _init_paths
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from datasets.cleargrasp.dataset import PoseDataset as PoseDataset_cleargrasp
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
import copy
import scipy.io as scio
import warnings

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cleargrasp', help='cleargrasp or ycb or linemod')
parser.add_argument('--dataset_root', type=str, default='/home/cxt/Documents/research/lf_perception/598-007-project/',
                    help='dataset root dir (''cleargrasp_preprocessed'' or ''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03,
                    help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default=2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default='', help='resume PoseNet model')
parser.add_argument('--resume_refinenet', type=str, default='', help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
opt = parser.parse_args()
bs = 1
#result_wo_refine_dir = 'experiments/eval_result/cleargrasp/Densefusion_wo_refine_result_dandan_fixed'
#result_wo_refine_dir = 'experiments/eval_result/cleargrasp/temp'
result_refine_dir = 'experiments/eval_result/cleargrasp/Densefusion_iterative_result'
#result_wo_refine_dir = './'
if not os.path.exists(result_wo_refine_dir):
    os.makedirs(result_wo_refine_dir)	


def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'ycb':
        opt.num_objects = 21  # number of object classes in the dataset
        opt.num_points = 1000  # number of points on the input pointcloud
        opt.outf = 'trained_models/ycb'  # folder to save trained models
        opt.log_dir = 'experiments/logs/ycb'  # folder to save logs
        opt.repeat_epoch = 1  # number of repeat times for one epoch training
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.outf = 'trained_models/linemod'
        opt.log_dir = 'experiments/logs/linemod'
        opt.repeat_epoch = 20
    elif opt.dataset == 'cleargrasp':
        opt.num_objects = 5
        opt.num_points = 500
        opt.outf = 'trained_models/cleargrasp'
        opt.log_dir = 'experiments/logs/cleargrasp'
        opt.repeat_epoch = 1
    else:
        print('Unknown dataset')
        return

    estimator = PoseNet(num_points=opt.num_points, num_obj=opt.num_objects)
    #estimator.cuda()
    refiner = PoseRefineNet(num_points=opt.num_points, num_obj=opt.num_objects)
    #refiner.cuda()

    if opt.resume_posenet != '':
        #estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet), map_location='cpu'))

    if opt.resume_refinenet != '':
        refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))
        opt.refine_start = True
        opt.decay_start = True
        opt.lr *= opt.lr_rate
        opt.w *= opt.w_rate
        opt.batch_size = int(opt.batch_size / opt.iteration)
        optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    # if opt.dataset == 'ycb':
    #     dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    # elif opt.dataset == 'linemod':
    #     dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans,
    #                                   opt.refine_start)
    # elif opt.dataset == 'cleargrasp':
    #     dataset = PoseDataset_cleargrasp('train', opt.num_points, True, opt.dataset_root, opt.noise_trans,
    #                                      opt.refine_start)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'cleargrasp':
        test_dataset = PoseDataset_cleargrasp('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False,
                                                 num_workers=opt.workers)

    opt.sym_list = test_dataset.get_sym_list()
    opt.num_points_mesh = test_dataset.get_num_points_mesh()
    #
    # print(
    #     '>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(
    #         len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_points_mesh, opt.sym_list)
    criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

    # best_test = np.Inf
    #
    # os.system('mkdir -p ' + opt.log_dir)
    # if opt.start_epoch == 1:
    #     for log in os.listdir(opt.log_dir):
    #         os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()

    for epoch in range(0, 1):
        # logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        # logger.info('Train time {0}'.format(
        #     time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        # train_count = 0
        # train_dis_avg = 0.0
        # if opt.refine_start:
        #     estimator.eval()
        #     refiner.train()
        # else:
        #     estimator.train()
        # optimizer.zero_grad()

        # for rep in range(opt.repeat_epoch):
        #     for i, data in enumerate(dataloader):
        #         points, choose, img, target, model_points, idx = data
        #         points, choose, img, target, model_points, idx = Variable(points).cuda(), \
        #                                                          Variable(choose).cuda(), \
        #                                                          Variable(img).cuda(), \
        #                                                          Variable(target).cuda(), \
        #                                                          Variable(model_points).cuda(), \
        #                                                          Variable(idx).cuda()
        #         pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        #         loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points,
        #                                                       opt.w, opt.refine_start)
        #
        #         if opt.refine_start:
        #             for ite in range(0, opt.iteration):
        #                 pred_r, pred_t = refiner(new_points, emb, idx)
        #                 dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx,
        #                                                                new_points)
        #                 dis.backward()
        #         else:
        #             loss.backward()
        #
        #         train_dis_avg += dis.item()
        #         train_count += 1
        #
        #         if train_count % opt.batch_size == 0:
        #             logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'.format(
        #                 time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch,
        #                 int(train_count / opt.batch_size), train_count, train_dis_avg / opt.batch_size))
        #             # print('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, int(train_count / opt.batch_size), train_count, train_dis_avg / opt.batch_size))
        #             optimizer.step()
        #             optimizer.zero_grad()
        #             train_dis_avg = 0
        #
        #         if train_count != 0 and train_count % 1000 == 0:
        #             # if epoch % 1 == 0 and train_count != 0 and train_count % 1000 == 0:
        #             if opt.refine_start:
        #                 torch.save(refiner.state_dict(), '{0}/pose_refine_current_model_obj.pth'.format(opt.outf))
        #             else:
        #                 torch.save(estimator.state_dict(), '{0}/pose_current_model_obj.pth'.format(opt.outf))
        #
        # print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()
        refiner.eval()
        # TODO: change dataloader or add loop to go through all instances in images, now randomly choose one from each image
        for j, data in enumerate(testdataloader, 0):
            my_result = []
            # TODO: change these variables with #instances in the last dimension
            my_result_wo_refine = np.zeros((1, 7))
            gt_pose = np.zeros((3, 4, 1))
            cls_indexes = np.zeros((1,))
            instance_ids = np.zeros((1,))

            points, choose, img, target, model_points, idx, gt_pose_r, gt_pose_t = data

            '''
            cloud, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                             Variable(choose).cuda(), \
                                                             Variable(img).cuda(), \
                                                             Variable(target).cuda(), \
                                                             Variable(model_points).cuda(), \
                                                             Variable(idx).cuda()
            '''

            cloud, choose, img, target, model_points, idx = Variable(points), \
                                                             Variable(choose), \
                                                             Variable(img), \
                                                             Variable(target), \
                                                             Variable(model_points), \
                                                             Variable(idx)

            pred_r, pred_t, pred_c, emb = estimator(img, cloud, choose, idx)
            #_, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, cloud, opt.w,
             #                                          opt.refine_start)

            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, opt.num_points, 1)

            pred_c = pred_c.view(bs, opt.num_points)
            how_max, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(bs * opt.num_points, 1, 3)
            points = cloud.view(bs * opt.num_points, 1, 3)

            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
            my_pred = np.append(my_r, my_t)
            my_result_wo_refine[0] = np.array(my_pred.tolist())
            cls_indexes[0] = idx.cpu().numpy()[0] + 1 # MATLAB indices starts from 1

            if opt.refine_start:
                for ite in range(0, opt.iteration):
                    '''
                    T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).\
                        repeat(opt.num_points, 1).contiguous().view(1, opt.num_points, 3)
                    '''
                    T = Variable(torch.from_numpy(my_t.astype(np.float32))).view(1, 3).\
                        repeat(opt.num_points, 1).contiguous().view(1, opt.num_points, 3)

                    my_mat = quaternion_matrix(my_r)
                    #R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                    R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).view(1, 3, 3)
                    my_mat[0:3, 3] = my_t

                    new_cloud = torch.bmm((cloud - T), R).contiguous()
                    pred_r, pred_t = refiner(new_cloud, emb, idx)
                    dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx,
                                                                   new_points)
                    pred_r = pred_r.view(1, 1, -1)
                    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                    my_r_2 = pred_r.view(-1).cpu().data.numpy()
                    my_t_2 = pred_t.view(-1).cpu().data.numpy()
                    my_mat_2 = quaternion_matrix(my_r_2)

                    my_mat_2[0:3, 3] = my_t_2

                    my_mat_final = np.dot(my_mat, my_mat_2)
                    my_r_final = copy.deepcopy(my_mat_final)
                    my_r_final[0:3, 3] = 0
                    my_r_final = quaternion_from_matrix(my_r_final, True)
                    my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                    my_pred = np.append(my_r_final, my_t_final)
                    my_r = my_r_final
                    my_t = my_t_final
                my_result.append(my_pred.tolist())

            #test_dis += dis.item()
            #logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(
             #   time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis))

            #test_count += 1
            gt_pose[:, :, 0] = np.hstack((gt_pose_r[0].numpy(), gt_pose_t[0].numpy().transpose()))
            scio.savemat('{0}/{1}.mat'.format(result_wo_refine_dir, '%04d' % j),
                         {'poses': my_result_wo_refine, 'gt_poses': gt_pose, 'cls_indexes': cls_indexes})
            if opt.refine_start:
                scio.savemat('{0}/{1}.mat'.format(result_refine_dir, '%04d' % j), {'poses': my_result, 'gt_poses': gt_pose, 'cls_indexes': idx.cpu().numpy()[0]})
            print("Finish No.{0} keyframe".format(j))

        #test_dis = test_dis / test_count
        #logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(
         #   time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))



        # if test_dis <= best_test:
        #     best_test = test_dis
        #     if opt.refine_start:
        #         torch.save(refiner.state_dict(),
        #                    '{0}/pose_refine_model_{1}_{2}_obj.pth'.format(opt.outf, epoch, test_dis))
        #     else:
        #         torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}_obj.pth'.format(opt.outf, epoch, test_dis))
        #     print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        # if best_test < opt.decay_margin and not opt.decay_start:
        #     opt.decay_start = True
        #     opt.lr *= opt.lr_rate
        #     opt.w *= opt.w_rate
        #     optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

        # if best_test < opt.refine_margin and not opt.refine_start:
        #     opt.refine_start = True
        #     # opt.batch_size = int(opt.batch_size / opt.iteration)
        #     optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
        #
        #     if opt.dataset == 'ycb':
        #         dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans,
        #                                   opt.refine_start)
        #     elif opt.dataset == 'linemod':
        #         dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans,
        #                                       opt.refine_start)
        #     elif opt.dataset == 'cleargrasp':
        #         dataset = PoseDataset_cleargrasp('train', opt.num_points, True, opt.dataset_root, opt.noise_trans,
        #                                          opt.refine_start)
        #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
        #     if opt.dataset == 'ycb':
        #         test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
        #     elif opt.dataset == 'linemod':
        #         test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0,
        #                                            opt.refine_start)
        #     elif opt.dataset == 'cleargrasp':
        #         test_dataset = PoseDataset_cleargrasp('test', opt.num_points, False, opt.dataset_root, 0.0,
        #                                               opt.refine_start)
        #     testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
        #                                                  num_workers=opt.workers)
        #
        #     opt.sym_list = dataset.get_sym_list()
        #     opt.num_points_mesh = dataset.get_num_points_mesh()
        #
        #     print(
        #         '>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(
        #             len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))
        #
        #     criterion = Loss(opt.num_points_mesh, opt.sym_list)
        #     criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)


if __name__ == '__main__':
    main()
