import json
import numpy as np
from numpy.linalg import inv
import os
import scipy.io as sio
from PIL import Image
from scipy.spatial.transform import Rotation as R

'''
obj_to_label={'cup-with-waves-train': 1,
 	      'flower-bath-bomb-train' : 2,
               'heart-bath-bomb-train' : 3,
               'square-plastic-bottle-train' : 4,
               'stemless-plastic-champagne-glass-train' : 5}

'''
obj_to_label={'cup-with-waves-val': 1,
	      'flower-bath-bomb-val' : 2,
              'heart-bath-bomb-val' : 3,
              'square-plastic-bottle-val' : 4,
              'stemless-plastic-champagne-glass-val' : 5}

root = '../cleargrasp-testing-validation/synthetic-val/'	
#root = '../cleargrasp-dataset-train/'
obj_folders = sorted(os.listdir(root))
factor_depth = 4000

for fold in obj_folders:
    label = obj_to_label[fold]
    os.makedirs(root + fold + '/meta-files/', exist_ok=True)
    # os.makedirs(root + fold + '/meta-files-temp/', exist_ok=True)
    json_dir = root + fold + '/json-files/'
    json_files = os.listdir(json_dir)
    
    img_dir = root + fold + '/segmentation-masks/'

    for f in sorted(json_files):
        print('Processing ' + f + ' from object ' + fold)
        with open(json_dir + '/' + f) as js: 
            data = json.load(js)
        meta = {}
        cam_matrix = np.array(data['camera']['world_pose']['matrix_4x4'])
        q = data['camera']['world_pose']["rotation"]["quaternion"]
        r = R.from_quat([q[1], q[2], q[3], q[0]])
        cam_matrix[:3, :3] = r.as_dcm()
        num_obj = len(data['variants']['masks_and_poses_by_pixel_value'])
        obj_keys = sorted(list(data['variants']['masks_and_poses_by_pixel_value'].keys()))

        cls_indices = np.ones([num_obj,1]) * label
        poses = []
        for key in obj_keys:
            obj_pose = data['variants']['masks_and_poses_by_pixel_value'][key]['world_pose']['matrix_4x4']
            cam_inv = np.eye(4)
            cam_inv[:3, :3] = np.transpose(np.array(cam_matrix)[:3, :3])
            cam_inv[:3, 3] = -np.matmul(cam_inv[:3, :3], np.array(cam_matrix)[:3, 3])
            pose_in_cam = np.matmul(cam_inv, obj_pose)
            if len(poses)==0:
                poses = pose_in_cam[:3]
            else:
                poses = np.dstack((poses, pose_in_cam[:3]))
        
        if len(np.shape(poses))==2:
            poses = np.expand_dims(poses, axis=2)
        meta['cls_indexes'] = cls_indices
        meta['poses'] = poses
        meta['factor_depth'] = factor_depth
        meta['instance_ids'] = [float(i) for i in obj_keys]
        name = f.split('-')[0]+'.mat'
        sio.savemat(root + fold + '/meta-files/'+ name, meta)
        
        
        # # temp data, with same content, saved as text for C++ reading ease
        # name_prefix = f.split('-')[0]
        # np.savetxt(root + fold + '/meta-files-temp/' + name_prefix+'cls_indexes.txt', cls_indices, fmt='%d')
        # with open(root + fold + '/meta-files-temp/' + name_prefix+'poses.txt', 'w') as outfile:
        #     for pose in poses:
        #         np.savetxt(outfile, np.flipud(np.transpose(pose)), fmt='%-7.5f')
                
        # a = 1
        
        # # read image size and save for C++ ease
        # im = Image.open(img_dir + name_prefix + '-segmentation-mask.png')
        # w, h = im.size
        # with open(root + fold + '/meta-files-temp/' + name_prefix+'imgsize.txt', 'w') as outfile:
        #     outfile.write(str(w))
        #     outfile.write('\n')
        #     outfile.write(str(h))
