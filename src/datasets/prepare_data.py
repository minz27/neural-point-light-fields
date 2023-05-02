import argparse
import os
import glob
import numpy as np
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils, range_image_utils
import imageio
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import pickle
from collections import defaultdict
from copy import deepcopy
import cv2
import open3d as o3d
import pandas as pd
from typing import List
from PIL import Image

def extract_object_labels(datadirs:List[str], saving_dir:str):
    for file_num,file in enumerate(datadirs):
        file_name = file.split('/')[-1].split('.')[0]
        tracking_info = {}
        print("Procesing %s"%file_name)
        camera_labels = {'npypath': [], 'xmin': [], 'xmax': [], 'ymin': [], 'ymax': [], 'motionFlag': []}
        dataset = tf.data.TFRecordDataset(file, compression_type='')
        for f_num, data in enumerate(tqdm(dataset)):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            for im in frame.images:
                camera_name = open_dataset.CameraName.Name.Name(im.name)
                if camera_name == 'FRONT_RIGHT':
                    camera_name = 'SIDE_LEFT'
                elif camera_name == 'SIDE_LEFT':
                    camera_name = 'FRONT_RIGHT'    
                npypath = os.path.join(saving_dir,file_name, 'custom_npy','%03d_%s.npy'%(f_num,camera_name))
                # for obj_label in frame.projected_lidar_labels[im.name-1].labels:
                for obj_label in frame.camera_labels[im.name-1].labels:
                    xmin = obj_label.box.center_x - 0.5*obj_label.box.length
                    ymin = obj_label.box.center_y - 0.5*obj_label.box.width
                    xmax = xmin + obj_label.box.length
                    ymax = ymin + obj_label.box.width
                    camera_labels['npypath'].append(npypath)
                    camera_labels['xmin'].append(xmin)
                    camera_labels['xmax'].append(xmax)
                    camera_labels['ymin'].append(ymin)
                    camera_labels['ymax'].append(ymax)
                    camera_labels['motionFlag'].append(0)
                
        df = pd.DataFrame.from_dict(camera_labels)
        #Sort df
        #Regex matches frame numbers like 000, 001, 002, etc
        df['frame_value'] = df['npypath'].str[-20:].str.extractall(r'(\d+)').values
        #Regex matches _FRONT.npy, _FRONT_LEFT.npy, _FRONT_RIGHT.npy, _SIDE_LEFT.npy, _SIDE_RIGHT.npy
        df['sorter'] = df['npypath'].str[-20:].str.extractall(r'(_[A-Z]*_?[A-Z]+.npy)').values
        df['sorter'] = df['sorter'] + '_' + df['frame_value']
        df = df.sort_values('sorter')
        df = df.drop(['sorter', 'frame_value'], axis = 1)
        csv_file_name = os.path.join(saving_dir, file_name,'custom_label.csv')
        df.to_csv(csv_file_name, index=False)    
  
def normalize8(I):
  mn = I.min()
  mx = I.max()

  mx -= mn

  I = ((I - mn)/mx) * 255
  return I.astype(np.uint8)

def extract_depth_maps(datadirs:List[str], saving_dir:str):
    for file_num,file in enumerate(datadirs):
        file_name = file.split('/')[-1].split('.')[0]
        if not os.path.isdir(os.path.join(saving_dir,file_name, 'depth_images')):   os.mkdir(os.path.join(saving_dir,file_name, 'depth_images'))
        print("Procesing %s"%file_name)
        dataset = tf.data.TFRecordDataset(file, compression_type='')
        for f_num, data in enumerate(tqdm(dataset)):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            (range_images, camera_projections, range_image_top_pose) = \
                frame_utils.parse_range_image_and_camera_projection(frame)
            range_image_cartesian = frame_utils.convert_range_image_to_cartesian(frame,
                                     range_images,
                                     range_image_top_pose,
                                     ri_index=0,
                                     keep_polar_features=False)                             
            for im in frame.images:
                camera_name = open_dataset.CameraName.Name.Name(im.name)
                depth_path = os.path.join(saving_dir,file_name, 'depth_images','%03d_%s.png'%(f_num,camera_name))
                extrinsic = np.reshape(frame.context.camera_calibrations[im.name-1].extrinsic.transform, [1,4,4]).astype(np.float32)
                camera_image_size = (frame.context.camera_calibrations[im.name-1].height, frame.context.camera_calibrations[im.name-1].width)
                
                ric_shape = range_image_cartesian[im.name].shape
                ric = np.reshape(range_image_cartesian[im.name], [1, ric_shape[0], ric_shape[1], ric_shape[2]])
                # print(camera_projections)
                # cp_shape = camera_projections[im.name].shape
                # cp = np.reshape(camera_projections[im.name], [1, cp_shape[0], cp_shape[1], cp_shape[2]])
                cp = camera_projections[im.name][0]
                cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
                cp_shape = cp_tensor.shape
                cp_tensor = np.reshape(cp_tensor, [1, cp_shape[0], cp_shape[1], cp_shape[2]])
                depth_image = range_image_utils.build_camera_depth_image(ric,
                             extrinsic,
                             cp_tensor,
                             camera_image_size,
                             im.name)
                depth_shape = depth_image.shape
                depth_image = np.reshape(depth_image, [depth_shape[1], depth_shape[2]])
                depth_image = normalize8(depth_image) 
                        
                imageio.imwrite(depth_path, depth_image, compress_level=3)          

def mask_images(datadirs:List[str]):
    raw_img_dir = os.path.join(datadirs, "images")
    masked_img_dir = os.path.join(datadirs, 'masked_images')
    if not os.path.isdir(masked_img_dir): os.mkdir(masked_img_dir)

    label_filepath = os.path.join(datadirs, "custom_label.csv")
    labels = pd.read_csv(label_filepath)
    labels['motionFlagBool'] = labels['motionFlag'].str[1:2].astype(int)
    labels['img_name'] = [x.split('/')[-1][:-4] for x in labels['npypath']]

    for img_name in labels['img_name'].unique():
        img_metadata = labels[labels['img_name'] == img_name]
        orig_img_path = os.path.join(raw_img_dir, img_name + '.png')
        masked_img_path = os.path.join(masked_img_dir, img_name + '.png')
        img = Image.open(orig_img_path)
        img_np = np.array(img)
        for index, row in img_metadata.iterrows():
            if row['motionFlagBool'] == 1:
                img_np[int(row['ymin']):int(row['ymax']), int(row['xmin']):int(row['xmax'])] = 0
        imageio.imwrite(masked_img_path, img_np, compress_level=3)
                            

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir')
    parser.add_argument('-task', '--task')
    args,_ = parser.parse_known_args()
    datadirs = args.datadir
    saving_dir = '/'.join(datadirs.split('/')[:-1])
    if '.tfrecord' not in datadirs:
        saving_dir = 1*datadirs
        datadirs = glob.glob(datadirs+'/*.tfrecord',recursive=True)
        datadirs = sorted([f for f in datadirs if '.tfrecord' in f])
        MULTIPLE_DIRS = True

    if not isinstance(datadirs,list):   datadirs = [datadirs]
    if not os.path.isdir(saving_dir):   os.mkdir(saving_dir)
    task = args.task
    if task == 'labels':
        extract_object_labels(datadirs, saving_dir)
    elif task == 'depth':
        extract_depth_maps(datadirs)    
    elif task == 'mask':
        # Pass the path of the scene directly here, will mask per scene for now
        mask_images(args.datadir)    