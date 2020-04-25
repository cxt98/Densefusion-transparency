import _init_paths
import argparse
import os
import copy
import random
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset
from datasets.cleargrasp.dataset import PoseDataset as PoseDataset_cleargrasp
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str,
                    default='/media/cravisjan97/New Volume/UMICH WINTER 2020/EECS 598-007/Project/cleargrasp-testing-validation/synthetic-val/', help='dataset root dir')
parser.add_argument('--model', type=str,
                    default='/media/cravisjan97/New Volume/UMICH WINTER 2020/EECS 598-007/Project/Densefusion-transparency/trained_models/cleargrasp/pose_model_16_0.15974816026976782_obj.pth', help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default='', help='resume PoseRefineNet model')
parser.add_argument('--model_path', type=str,
                    default='/media/cravisjan97/New Volume/UMICH WINTER 2020/EECS 598-007/Project/cleargrasp-3d-models-fixed')
opt = parser.parse_args()

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
img_width = 1920
img_height = 1080
img_step = 40
border_list = np.arange(0, img_width + img_step + 1, img_step)
# border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(img_width)] for j in range(img_height)])
ymap = np.array([[i for i in range(img_width)] for j in range(img_height)])
cam_cx = 960
cam_cy = 540
cam_fx = 1386.4
cam_fy = 1386.4
cam_scale = 1.0  # 1.0 for exr, 4000 for png
num_obj = 5
# img_width = 480
# img_length = 640
num_points = 500
minimum_num_pt = 50
num_points_mesh = 500
num_pt_mesh_small = 500
num_pt_mesh_large = 2600
iteration = 2
bs = 1
refine = False
dataset_config_dir = 'datasets/cleargrasp/dataset_config'
#ycb_toolbox_dir = 'YCB_Video_toolbox'
result_wo_refine_dir = 'experiments/eval_result/cleargrasp/Densefusion_wo_refine_result_all_instances/'
#result_refine_dir = 'experiments/eval_result/cleargrasp/Densefusion_iterative_result_all_instances'
if not os.path.exists(result_wo_refine_dir):
    os.makedirs(result_wo_refine_dir)	


# borrowed from dataset。py
def get_bbox(label, img_width, img_length, border_list):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    #print(np.shape(np.where(rows)))
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


# borrowed from dataset.py
def list2realpath(ls, param, suffix):
    object_path = ls.split('/')[0] + '/' + ls.split('/')[1]
    modality_path = ls.split('/')[-2]
    file_id = ls.split('/')[-1]
    #return '{0}/{1}/{2}/{3}{4}'.format(opt.dataset_root, object_path + '/' + modality_path, param, file_id, suffix)
    return '{0}/{1}/{2}/{3}{4}'.format(opt.dataset_root, modality_path, param, file_id, suffix)


# borrowed from dataset.py
def exr_loader(EXR_PATH, ndim=3):
    # use opencv library instead
    img = cv2.imread(EXR_PATH, cv2.IMREAD_UNCHANGED)
    if ndim == 1:
        img = img[:, :, 0]
    return img
    """Loads a .exr file as a numpy array
    Args:
        EXR_PATH: path to the exr file
        ndim: number of channels that should be in returned array. Valid values are 1 and 3.
                        if ndim=1, only the 'R' channel is taken from exr file
                        if ndim=3, the 'R', 'G' and 'B' channels are taken from exr file.
                            The exr file must have 3 channels in this case.
    Returns:
        numpy.ndarray (dtype=np.float32): If ndim=1, shape is (height x width)
                                          If ndim=3, shape is (3 x height x width)
    """
    import OpenEXR
    import Imath
    exr_file = OpenEXR.InputFile(EXR_PATH)
    cm_dw = exr_file.header()['dataWindow']
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    if ndim == 3:
        # read channels indivudally
        allchannels = []
        for c in ['R', 'G', 'B']:
            # transform data to numpy
            channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
            channel.shape = (size[1], size[0])
            allchannels.append(channel)
        # create array and transpose dimensions to match tensor style
        exr_arr = np.array(allchannels).transpose((0, 1, 2))
        return exr_arr
    if ndim == 1:
        # transform data to numpy
        channel = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        exr_arr = np.array(channel)
        return exr_arr

# dataset = PoseDataset_cleargrasp('test', 500, False, opt.dataset_root, 0.0, False)
# criterion = Loss(dataset.get_num_points_mesh(), dataset.get_sym_list())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
estimator = PoseNet(num_points=num_points, num_obj=num_obj)
#estimator.cuda()
#estimator.load_state_dict(torch.load(opt.model))
estimator.load_state_dict(torch.load(opt.model))
estimator.to(device)
estimator.eval()


refiner = PoseRefineNet(num_points=num_points, num_obj=num_obj)
#refiner.cuda()
if refine:
    refiner.load_state_dict(torch.load(opt.refine_model))
refiner.to(device)
refiner.eval()

testlist = []
input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))  # TODO: now only trained on one object (cup)
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]
    testlist.append(input_line)
input_file.close()
print(len(testlist))

class_file = open('{0}/classes.txt'.format(dataset_config_dir))
class_id = 1
cld = {}
while 1:
    class_input = class_file.readline()
    if not class_input:
        break
    class_input = class_input[:-1]

    input_file = open('{0}/{1}.xyz'.format(opt.model_path, class_input))
    cld[class_id] = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1]
        input_line = input_line.split(' ')
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    input_file.close()
    cld[class_id] = np.array(cld[class_id])
    class_id += 1
trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)

for now in range(0, len(testlist)):
    img = Image.open(list2realpath(testlist[now], 'rgb-imgs', '-rgb.jpg'))
    # img = Image.open('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
    # read depth from exr file
    depth = exr_loader(list2realpath(testlist[now], 'depth-imgs-rectified', '-depth-rectified.exr'), ndim=1)
    depth = np.fliplr(depth)
    # depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root, testlist[now])))
    # posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
    meta = scio.loadmat(list2realpath(testlist[now], 'meta-files', '.mat'))
    # read segmentation from exr file
    label = exr_loader(list2realpath(testlist[now], 'variant-masks', '-variantMasks.exr'), ndim=1)
    label = np.fliplr(label)

    mask_back = ma.getmaskarray(ma.masked_equal(label, 0))
    obj = meta['cls_indexes'].flatten().astype(np.int32)
    #instance_id = meta['instance_ids'].flatten().astype(np.int32)
    instance_id = np.unique(label).tolist()[1:]
    my_result_wo_refine = []
    my_result = []
    valididx = []
    gt_poses = []
    for idx in range(len(instance_id)):
        gt_pose = np.zeros((3, 4, 1))
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(label, instance_id[idx]))
        mask = mask_label * mask_depth
        if len(mask.nonzero()[0]) < minimum_num_pt:
            continue
        valididx.append(idx)
        img_width, img_height = depth.shape
        img_step = 40
        border_list = np.arange(0, img_width + img_step + 1, img_step)
        border_list[0] = -1

        rmin, rmax, cmin, cmax = get_bbox(mask_label, img_width, img_height, border_list)
        img1 = np.fliplr(np.array(img))
        img1 = np.transpose(img1[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

        seed = random.choice(testlist)
        back = np.array(trancolor(Image.open(list2realpath(seed, 'rgb-imgs', '-rgb.jpg')).convert("RGB")))
        back = np.transpose(back, (2, 0, 1))[:, rmin:rmax, cmin:cmax]
        img_masked = back * mask_back[rmin:rmax, cmin:cmax] + img1

        img_masked = img_masked + np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)

        target_r = meta['poses'][:, :, idx][:, 0:3]
        target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])

        add_t = np.array([random.uniform(-0.03, 0.03) for i in range(3)])

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) > num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:num_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        cam_scale = 1.0  # cleargrasp saves depth in exr format in float, no scale
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)

        dellist = [j for j in range(0, len(cld[obj[idx]]))]
        if refine:
            dellist = random.sample(dellist, len(cld[obj[idx]]) - num_pt_mesh_large)
        else:
            dellist = random.sample(dellist, len(cld[obj[idx]]) - num_pt_mesh_small)
        model_points = np.delete(cld[obj[idx]], dellist, axis=0)

        target = np.dot(model_points, target_r.T)
        target = np.add(target, target_t)
        #print(np.shape(target_t))

        cloud = torch.from_numpy(cloud.astype(np.float32))
        cloud = cloud.view(1, num_points, 3)
        choose = torch.LongTensor(choose.astype(np.int32))
        img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
        img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])
        target = torch.from_numpy(target.astype(np.float32))
        model_points = torch.from_numpy(model_points.astype(np.float32))
        index = torch.LongTensor([int(obj[idx]) - 1])

        '''
        model_points = Variable(model_points).cuda()
        cloud = Variable(cloud).cuda()
        choose = Variable(choose).cuda()
        img_masked = Variable(img_masked).cuda()
        index = Variable(index).cuda()
        target = Variable(target).cuda()
        '''
        model_points = Variable(model_points).to(device)
        cloud = Variable(cloud).to(device)
        choose = Variable(choose).to(device)
        img_masked = Variable(img_masked).to(device)
        index = Variable(index).to(device)
        target = Variable(target).to(device)

        pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
        #_, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, index, cloud,0.015,
         #                                          False)
        #print(dis.item())
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

        pred_c = pred_c.view(bs, num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_points, 1, 3)
        points = model_points.view(bs * num_points, 1, 3)

        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        my_pred = np.append(my_r, my_t)
        my_result_wo_refine.append(my_pred.tolist())
        #gt_poses.append(np.vstack((target_r[0], target_t[0])).transpose())
        #gt = np.concatenate([target_r, target_t.transpose()], axis=1)
        #gt = gt[:, :,np.newaxis]
        gt_pose[:, :, 0] = np.hstack((target_r, target_t.transpose()))
        #print(np.shape(gt))
        if len(gt_poses)!=0:
            gt_poses = np.concatenate([gt_poses, gt_pose], axis=2)
        else:
            gt_poses = gt_pose

    scio.savemat('{0}/{1}.mat'.format(result_wo_refine_dir, '%04d' % now),
                 {'poses': my_result_wo_refine, 'gt_poses': gt_poses, 'cls_indexes': obj[valididx]})
    #scio.savemat('{0}/{1}.mat'.format(result_refine_dir, '%04d' % now), {'poses': my_result})
    print("Finish No.{0} keyframe".format(now))
