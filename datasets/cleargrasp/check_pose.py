import json
import numpy as np
from numpy.linalg import inv
import os
from PIL import Image
from scipy.spatial.transform import Rotation as R
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path and sys.version_info[0] == 3:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
else:
    import cv2
import math
import scipy.io as sio

mode = 'val'  # 'val'

objs = ['heart-bath-bomb', 'square-plastic-bottle', 'cup-with-waves', 'flower-bath-bomb', 'stemless-plastic-champagne-glass']

obj_modelpath = '../cleargrasp-3d-models-fixed/'

if mode == 'train':
    root = '../cleargrasp-dataset-train/'
else:
    root = '../cleargrasp-testing-validation/synthetic-val/'
obj_folders = sorted(os.listdir(root))
x_axis_rads = 1.2112585306167603
y_axis_rads = 0.7428327202796936
sample_point = 5000
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

check_instance_consistency = False
check_projection = False

for obj in objs:
    json_dir = root + obj + '-' + mode + '/json-files/'
    mat_dir = root + obj + '-' + mode + '/meta-files/'  # check correctness of mat files
    
    # check file consistency
    if check_instance_consistency:
        print('checking instance consistency')
        for f in sorted(os.listdir(json_dir)):
            # print('Processing ' + f + ' from object ' + obj)
            with open(json_dir + '/' + f) as js: 
                data = json.load(js)
                inst_json = sorted(np.array([int(n) for n in data['variants']['masks_and_poses_by_pixel_value'].keys()]))
            instance_segfold = root + obj + '-' + mode + '/variant-masks/'
            instance_seg = cv2.imread(instance_segfold + f.split('-')[0] + '-variantMasks.exr', cv2.IMREAD_UNCHANGED)[:, :, 0]     
            inst_varimask = sorted(np.delete(np.unique(instance_seg.flatten()), 0).astype('int'))
            
            if not np.array_equal(inst_json, inst_varimask):
                print(f + ' from object ' + obj)
                print('instances in json:')
                print(inst_json)
                print('instances in seg mask:')
                print(inst_varimask)
    
    # project objects to 2d
    # read poses from json files and use obj files extracted from dae files, calculate 2d projections and overlay to rgb images
    if check_projection:
        print('checking 2d projection')
        objfile = obj_modelpath + obj + '.obj'
        modelpoints = []
        with open(objfile, 'r') as f:
            for line in f:
                info = line.split(' ')
                if info[0] == 'v':
                    modelpoints.append([float(info[1]), float(info[2]), float(info[3])])
        modelpoints = np.array(modelpoints)
        # if modelpoints.shape[0] > sample_point:
        #     modelpoints = modelpoints[np.random.permutation(np.arange(modelpoints.shape[0]))[:sample_point]]
        modelpoints_aug = np.vstack((modelpoints.transpose(), np.ones((1, modelpoints.shape[0]))))
        
        count = 0
        
        for f in sorted(os.listdir(json_dir)):
            print('Processing ' + f + ' from object ' + obj)
            with open(json_dir + '/' + f) as js: 
                data = json.load(js)
            cam_matrix = np.array(data['camera']['world_pose']['matrix_4x4'])
            q = data['camera']['world_pose']["rotation"]["quaternion"]
            r = R.from_quat([q[1], q[2], q[3], q[0]])
            cam_matrix[:3, :3] = r.as_dcm()
            num_obj = len(data['variants']['masks_and_poses_by_pixel_value'])
            obj_keys = sorted(list(data['variants']['masks_and_poses_by_pixel_value'].keys()))
            
            rgb_img = cv2.imread(root + obj + '-' + mode + '/rgb-imgs/' + f.split('-')[0] + '-rgb.jpg')
            rgb_img_fliplr = cv2.flip(rgb_img, 1)
            
            for i, key in enumerate(obj_keys):
                obj_pose = data['variants']['masks_and_poses_by_pixel_value'][key]['world_pose']['matrix_4x4']
                cam_inv = np.eye(4)
                cam_inv[:3, :3] = np.transpose(np.array(cam_matrix)[:3, :3])
                cam_inv[:3, 3] = -np.matmul(cam_inv[:3, :3], np.array(cam_matrix)[:3, 3])
                pose_in_cam = np.matmul(cam_inv, obj_pose)
                # print(pose_in_cam)
                
                transformed_points_aug = np.matmul(pose_in_cam, modelpoints_aug)
                x_z = np.divide(transformed_points_aug[0], transformed_points_aug[2])
                y_z = np.divide(transformed_points_aug[1], transformed_points_aug[2])
                
                img_height, img_width, _ = rgb_img_fliplr.shape
                cx = img_width / 2
                cy = img_height / 2
                fx = cx / math.tan(x_axis_rads / 2)
                fy = cy / math.tan(y_axis_rads / 2)
                
                us = fx * x_z + cx
                vs = fy * y_z + cy
                
                for (u, v) in zip(us, vs):
                    if 1 < int(u) < img_width and 1 < int(v) < img_height:
                        cv2.circle(rgb_img_fliplr, (int(u), int(v)), 1, color=colors[i])
            
            cv2.imwrite(root + obj + f.split('-')[0] + '-rgb-project.jpg', rgb_img_fliplr)
            # if count == 5:
            #     break
            count += 1
        
# correctness of mat files, check
test_objs = ['heart-bath-bomb', 'square-plastic-bottle', 'stemless-plastic-champagne-glass']
testlist_file = '/home/cxt/Documents/research/lf_perception/598-007-project/DenseFusion/datasets/cleargrasp/dataset_config/test_data_list.txt'
testlist = []
with open(testlist_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        folder = line[:line.rfind('/')]
        index = line[line.rfind('/'):-1]
        testlist.append(root + folder + '/rgb-imgs' + index + '-rgb.jpg')

obj_points = {}
for obj in test_objs:
    objfile = obj_modelpath + obj + '.obj'
    modelpoints = []
    with open(objfile, 'r') as f:
        for line in f:
            info = line.split(' ')
            if info[0] == 'v':
                modelpoints.append([float(info[1]), float(info[2]), float(info[3])])
    modelpoints = np.array(modelpoints)
    obj_points[obj] = np.vstack((modelpoints.transpose(), np.ones((1, modelpoints.shape[0]))))

gt_mat_folder = '/home/cxt/Documents/research/lf_perception/598-007-project/DenseFusion/experiments/eval_result/cleargrasp/Densefusion_wo_refine_result_dandan_fixed/'
for i, f in enumerate(testlist):
    print('Processing ' + str(i + 1) + 'th image')
    meta = sio.loadmat(gt_mat_folder + '%04d.mat' % i)
    rgb_img = cv2.imread(f)
    rgb_img_fliplr = cv2.flip(rgb_img, 1)
    
    for obj in test_objs:
        if f.find(obj) != -1:
            modelpoints_aug = obj_points[obj]
            break
    
    for i in range(len(meta['cls_indexes'])):
        pose_in_cam = meta['gt_poses'][:, :, i]
        pose_in_cam[:3, :3] = pose_in_cam[:3, :3].transpose()
        
        transformed_points_aug = np.matmul(pose_in_cam, modelpoints_aug)
        x_z = np.divide(transformed_points_aug[0], transformed_points_aug[2])
        y_z = np.divide(transformed_points_aug[1], transformed_points_aug[2])
        
        img_height, img_width, _ = rgb_img_fliplr.shape
        cx = img_width / 2
        cy = img_height / 2
        fx = cx / math.tan(x_axis_rads / 2)
        fy = cy / math.tan(y_axis_rads / 2)
        
        us = fx * x_z + cx
        vs = fy * y_z + cy
        
        for (u, v) in zip(us, vs):
            if 1 < int(u) < img_width and 1 < int(v) < img_height:
                cv2.circle(rgb_img_fliplr, (int(u), int(v)), 1, color=colors[i])
    
    cv2.imwrite(root + obj + '-' + f[f.rfind('/')+1:], rgb_img_fliplr)
        
    # save xyz file for pose checking
    # xyz_dir = root + fold + '/pointclouds/'
    # os.system('mkdir -p ' + xyz_dir)
    # depthimg_dir = root + fold + '/depth-imgs-rectified/'
    # for f in sorted(os.listdir(depthimg_dir)):
    #     depthimg = cv2.imread(depthimg_dir + '/' + f, cv2.IMREAD_UNCHANGED)[:, :, 0]
    #     img_height, img_width = depthimg.shape
    #     cx = img_width / 2
    #     cy = img_height / 2
    #     fx = cx / math.tan(x_axis_rads / 2)
    #     fy = cy / math.tan(y_axis_rads / 2)
    #     u, v = np.meshgrid(np.arange(1, img_width + 1), np.arange(1, img_height + 1))
    #     x = np.multiply(depthimg, u - cx) / fx
    #     y = np.multiply(depthimg, v - cy) / fy
    #     xyz_array = np.vstack((x.flatten(), y.flatten(), depthimg.flatten())).transpose()
    #     np.savetxt(xyz_dir + f.split('-')[0]+'.xyz', xyz_array, fmt='%.6f')
    #     break
