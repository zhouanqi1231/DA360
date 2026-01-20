from __future__ import absolute_import, division, print_function
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from glob import glob
import argparse
import tqdm
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import cv2
import time

import networks
from datasets import Real
from saver import Saver

import open3d as o3d # You may need to pip install open3d

def save_point_cloud(depth, rgb, output_path):
    """
    Converts equirectangular depth and RGB to a 3D point cloud.
    depth: numpy array (H, W)
    rgb: numpy array (H, W, 3) - scaled 0-1
    """
    h, w = depth.shape
    
    # Create coordinate grids
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert to spherical angles
    theta = (u / w - 0.5) * 2 * np.pi
    phi = (0.5 - v / h) * np.pi
    
    # Calculate Cartesian coordinates
    x = depth * np.cos(phi) * np.sin(theta)
    y = depth * np.sin(phi)
    z = depth * np.cos(phi) * np.cos(theta)
    
    # Stack and reshape
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3)
    
    # Filter out invalid points (e.g., zero depth or infinity)
    mask = np.isfinite(points).all(axis=1) & (depth.flatten() > 0)
    points = points[mask]
    colors = colors[mask]

    # Create Open3D object and save
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pcd)

parser = argparse.ArgumentParser(description="Test Depth Anything in 360°")

parser.add_argument("--path", default="./data/images", type=str, help="path to test on.")

parser.add_argument("--model_path", type=str, help="path of model to load")
parser.add_argument("--net", type=str, default=None, help="model to use")
parser.add_argument('--model_name', type=str, default='DA360')

args = parser.parse_args()


def sort_key(s):
    import re
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', s)]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = torch.load(args.model_path)
    
    # data
    if 'net' not in model_dict:
        model_dict['net'] = 'DA360'
    if 'dinov2_encoder' not in model_dict:
        model_dict['dinov2_encoder'] = 'vits'
    if 'height' not in model_dict:
        model_dict['height'] = 518
    if 'width' not in model_dict:
        model_dict['width'] = 1036
    if args.net:
        model_dict['net'] = args.net
    
    # model
    Net = getattr(networks, model_dict['net'])
    model = Net(model_dict['height'], model_dict['width'], dinov2_encoder = model_dict['dinov2_encoder'])
    
    model.to(device)
    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict}, strict=False)
    model.eval()
        
    args.height = model_dict['height']
    args.width = model_dict['width']
    dataset = Real(args.height, args.width)
    dataset.root_dir = args.path
    rgb_list = glob(os.path.join(dataset.root_dir, '*.png')) + glob(os.path.join(dataset.root_dir, '*.jpg'))
    rgb_list = sorted(rgb_list, key=sort_key)
    dataset.rgb_list = rgb_list
    
    data_loader = data.DataLoader(dataset, 1, False, pin_memory=True, drop_last=False)
    pbar = tqdm.tqdm(data_loader)
    pbar.set_description("Testing")

    saver = Saver(os.path.join(args.model_path[:-4], "results"))

    for idx, inputs in enumerate(pbar):
        equi_inputs = inputs["normalized_rgb"].to(device)
        with torch.no_grad():
            start = time.time()
            outputs = model(equi_inputs)
            end = time.time()
        pred_disp = outputs["pred_disp"].detach().cpu()

        pred_depth = 1/pred_disp
        pred_depth = pred_depth/pred_depth.min()

        name = os.path.basename(rgb_list[idx])[:-4]

        saver.save_pred_samples(inputs["rgb"], pred_depth, name, args.model_name)

        # 1. Prepare data for point cloud
        # We use the raw RGB and the predicted depth
        depth_np = pred_depth.squeeze().cpu().numpy()
        
        # Prepare RGB: inputs["rgb"] is likely (1, 3, H, W) and 0-255 or 0-1
        rgb_np = inputs["rgb"].squeeze().permute(1, 2, 0).cpu().numpy()
        if rgb_np.max() > 1.0: rgb_np /= 255.0 # Normalize if needed

        # 2. Define output path
        name = os.path.basename(rgb_list[idx])[:-4]
        pc_path = os.path.join(args.model_path[:-4], "results", f"{name}.ply")
        
        # 3. Save
        save_point_cloud(depth_np, rgb_np, pc_path)

if __name__ == "__main__":
    main()
