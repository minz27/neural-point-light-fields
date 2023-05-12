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
from typing import List, Dict
from PIL import Image
import open3d as o3d
from pathlib import Path

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

def depth_read(filename:str)->np.array:
    # loads depth map D from png file
    # and returns it as a numpy array

    depth_png = np.array(Image.open(filename), dtype=int)
    assert(np.max(depth_png) > 255)
    depth = depth_png.astype(float) / 256.    
    depth[depth_png == 0] = -1.
    return depth

def read_calib_file(filepath:str)->Dict:
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def project_depth_map(depth_map:np.array, P:np.array)->o3d.geometry.PointCloud():
    height, width = depth_map.shape
    jj = np.tile(range(width), height)
    ii = np.repeat(range(height), width)
    xx = (jj - P[0,2])/P[0,0]
    yy = (ii - P[1,2])/P[1,1]
    length = height*width
    z = depth_map.reshape(length)
    pcd = np.dstack((xx * z, yy * z, z)).reshape((length, 3))
    pcd_o3d = o3d.geometry.PointCloud()  # create a point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    return pcd_o3d

def extract_point_clouds(datadir:Path)->None:
    depth_dir = datadir / 'image_03'
    pcl_dir = datadir / 'point_cloud'
    if not os.path.isdir(pcl_dir): os.mkdir(pcl_dir)        

    calib = read_calib_file(Path(datadir) / 'calib_cam_to_cam.txt')
    projection_matrix = calib['P_rect_03'].reshape(3,4)       
    P = calib['K_03'].reshape(3,3)
    R_z_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]) # rotation matrix for 180 degrees around z-axis
    R_x_90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) # rotation matrix for 90 degrees around x-axis
    R_y_90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]) # rotation matrix for 90 degrees around y-axis
    T = calib['T_03']

    depth_images = depth_dir.glob('*.png') 
    for fpath in depth_images:
        depth_map = depth_read(fpath)
        frame_name = fpath.stem
        pcd_o3d = project_depth_map(depth_map, P)

        # Convert to Waymo coordinates
        pcd_o3d = pcd_o3d.rotate(R_z_180, center=(0, 0, 0))
        pcd_o3d = pcd_o3d.rotate(R_y_90, center=(0, 0, 0))
        pcd_o3d = pcd_o3d.rotate(R_x_90, center=(0, 0, 0))     
        pcd_o3d = pcd_o3d.translate(T)

        #Save point clouds
        o3d.io.write_point_cloud(str(pcl_dir / f"{frame_name}.ply"), pcd_o3d)       

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir')
    parser.add_argument('-task', '--task')
    args,_ = parser.parse_known_args()
    datadirs = args.datadir
    saving_dir = '/'.join(datadirs.split('/')[:-1])

    # if not isinstance(datadirs,list):   datadirs = [datadirs]
    # if not os.path.isdir(saving_dir):   os.mkdir(saving_dir)
    task = args.task
    if task == 'labels':
        extract_object_labels(datadirs, saving_dir)
    elif task == 'pcl':
        extract_point_clouds(Path(args.datadir))    
    elif task == 'mask':
        mask_images(args.datadir)    