import cv2
import numpy as np
import torch
import torch.utils.data as data
import os


def medical_image_loader(img_path, image_size=(512, 512)):
    """
    加载医学图像并进行预处理
    返回: (processed_img, original_img)
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    
    original_img = img.copy()
    img = cv2.resize(img, image_size)
    
    # 标准化处理，与训练时保持一致
    processed_img = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    
    return processed_img, original_img


class MedicalImageFolder(data.Dataset):
    """
    医学图像数据集类，支持你的数据结构
    """
    
    def __init__(self, cfg, data_path):
        self.cfg = cfg
        self.data_path = data_path
        self.image_size = cfg.TEST.IMAGE_SIZE
        
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        self.image_paths = []
        self._collect_images()
        
        print(f"找到 {len(self.image_paths)} 张图像")
    
    def _collect_images(self):
        """收集所有图像文件"""
        if os.path.isfile(self.data_path):
            self.image_paths.append(self.data_path)
        elif os.path.isdir(self.data_path):
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in self.image_extensions):
                        self.image_paths.append(os.path.join(root, file))
        else:
            raise ValueError(f"路径不存在: {self.data_path}")
        
        self.image_paths.sort()
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        return None, None, None, img_path
    
    def __len__(self):
        return len(self.image_paths)