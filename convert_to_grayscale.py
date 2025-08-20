#!/usr/bin/env python3
"""
批量将DRIVE训练集和测试集图像转换为增强的3通道灰度图像。
输出目录: 
- dataset/DRIVE/training/images_enhanced/
- dataset/DRIVE/test/images_enhanced/
"""

import os
import cv2
import numpy as np
from PIL import Image
import shutil

def process_and_enhance_images(input_dir, output_dir):
    """
    批量转换指定目录的图像为增强的3通道灰度图像。
    """
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return False
    
    # 清理并创建输出目录
    if os.path.exists(output_dir):
        print(f"⚠️  输出目录已存在，正在清理: {output_dir}")
        shutil.rmtree(output_dir)
    
    print(f"✅ 创建输出目录: {output_dir}")
    os.makedirs(output_dir)
    
    # 初始化CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 获取图像文件列表
    try:
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))]
        if not image_files:
            print(f"❌ 在 {input_dir} 中未找到图像文件。")
            return False
    except OSError as e:
        print(f"❌ 读取目录 {input_dir} 时出错: {e}")
        return False

    print(f"\n总共找到 {len(image_files)} 张图像，开始处理...")
    processed_count = 0

    # 遍历并处理所有图像
    for filename in image_files:
        try:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with Image.open(input_path) as img:
                # 先转换为灰度图
                grayscale_pil = img.convert('L')
            
            # 转换为OpenCV格式并应用CLAHE增强
            grayscale_cv = np.array(grayscale_pil)
            enhanced_cv = clahe.apply(grayscale_cv)
            
            # 将增强后的灰度图转换为3通道图像
            enhanced_3ch = cv2.cvtColor(enhanced_cv, cv2.COLOR_GRAY2BGR)
            
            # 保存为3通道TIFF图像
            cv2.imwrite(output_path, enhanced_3ch)

            processed_count += 1
            print(f"  处理并保存: {filename}")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    print(f"\n🎉 处理完成！成功处理 {processed_count}/{len(image_files)} 张图像。")

    # 验证转换结果
    if processed_count > 0:
        print("\n🔍 开始验证结果...")
        first_image_path = os.path.join(output_dir, image_files[0])
        
        # 使用OpenCV读取图像并检查通道数
        img = cv2.imread(first_image_path)
        if img is None:
            print(f"  ❌ 无法读取图像: {first_image_path}")
        else:
            channels = img.shape[2] if len(img.shape) == 3 else 1
            print(f"  验证图像 '{image_files[0]}' 的通道数: {channels}")
            if channels == 3:
                print("  ✅ 验证成功，图像为3通道。")
            else:
                print(f"  ❌ 验证失败，图像为 {channels} 通道，不是3通道。")
    
    return True

if __name__ == '__main__':
    base_dir = "/mnt/data/lijianfei/DDRNet_SemanticSegmentation"
    
    # --- 处理训练集 ---
    print("🚀 开始处理训练集图像...")
    train_input_dir = os.path.join(base_dir, "dataset/DRIVE/training/images")
    train_output_dir = os.path.join(base_dir, "dataset/DRIVE/training/images_enhanced")
    train_success = process_and_enhance_images(train_input_dir, train_output_dir)
    if train_success:
        print("✅ 训练集处理成功。")
    else:
        print("❌ 训练集处理失败。")

    print("\n" + "="*50 + "\n")

    # --- 处理测试集 ---
    print("🚀 开始处理测试集图像...")
    test_input_dir = os.path.join(base_dir, "dataset/DRIVE/test/images")
    test_output_dir = os.path.join(base_dir, "dataset/DRIVE/test/images_enhanced")
    test_success = process_and_enhance_images(test_input_dir, test_output_dir)
    if test_success:
        print("✅ 测试集处理成功。")
    else:
        print("❌ 测试集处理失败。")
