from models.ddrnet import getDDRNet
from utils.config import get_cfg
import torch
import argparse
from torch.nn import Sigmoid
import numpy as np
from matplotlib import pyplot as plt
from utils.medical_data_utils import MedicalImageFolder, medical_image_loader
import os



def visualize(config, weight_path, data_path):

    model = getDDRNet(cfg=config)
    model.load_state_dict(torch.load(weight_path, map_location=config.TEST.DEVICE))
    model = model.cuda() if config.TEST.DEVICE=='cuda' else model
    model = model.eval()

    dataset = MedicalImageFolder(cfg=config, data_path=data_path)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.TEST.NUM_WORKERS)
    
    for i, img_path_batch in enumerate(test_loader):
        # 处理图像并进行预测
        img_path = img_path_batch[0]  # batch中的第一个路径
        img, original_img = medical_image_loader(img_path, image_size=config.TEST.IMAGE_SIZE)
        
        print(f"\n正在处理: {img_path}")
        
        # 转换为张量并添加batch维度
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.cuda() if config.TEST.DEVICE=='cuda' else img
            
        with torch.no_grad():
            pred = model(img)
            
        # 处理模型输出
        if isinstance(pred, list):
            pred = pred[0]  # 取主要输出
            
        pred = Sigmoid()(pred).cpu().numpy()
        
        # 处理预测结果形状
        if len(pred.shape) == 4:
            pred = pred[0, 0]  # [batch, channel, height, width] -> [height, width]
        elif len(pred.shape) == 3:
            pred = pred[0]  # [channel, height, width] -> [height, width]
            
        # 二值化
        pred_binary = pred.copy()
        pred_binary[pred_binary >= 0.5] = 1
        pred_binary[pred_binary < 0.5] = 0

        # 按照原始数据结构创建输出目录
        original_path = img_path
        
        # 解析路径结构: /path/to/高血压标签/类别/病人文件夹/图像文件
        path_parts = original_path.split(os.sep)
        
        # 找到高血压标签目录的位置
        try:
            base_idx = path_parts.index('高血压标签')
            category = path_parts[base_idx + 1]  # 高血压 或 健康
            patient_folder = path_parts[base_idx + 2]  # 病人文件夹
        except (ValueError, IndexError):
            # 如果路径结构不符合预期，使用相对路径
            category = os.path.basename(os.path.dirname(os.path.dirname(original_path)))
            patient_folder = os.path.basename(os.path.dirname(original_path))
        
        # 创建输出目录结构
        output_base_dir = "output_predictions"
        patient_output_dir = os.path.join(output_base_dir, category, patient_folder)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 保存分割掩码
        pred_save_path = os.path.join(patient_output_dir, f"{base_name}_mask.png")
        plt.figure(figsize=(8, 8))
        plt.imshow(pred_binary, cmap='gray')
        plt.axis('off')
        plt.savefig(pred_save_path, bbox_inches='tight', dpi=150, pad_inches=0, 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"分割掩码已保存: {pred_save_path}")
        print(f"预测范围: [{pred.min():.3f}, {pred.max():.3f}]")
        print("-" * 60)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical image segmentation inference')
    parser.add_argument('--cfg',
                        help='experiment config file address',
                        default="configs/ddrnet_DRIVE.yaml",
                        type=str)
    parser.add_argument('--weight',
                        help='path to the trained weights',
                        default="weights/best_loss.pth",
                        type=str)
    parser.add_argument('--data',
                        help='path to medical image data directory',
                        default='/mnt/data/lijianfei/高血压标签',
                        type=str)                    
    args = parser.parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    visualize(config=cfg, weight_path=args.weight, data_path=args.data)