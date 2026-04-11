import pickle
import os
from nuscenes.nuscenes import NuScenes

# ================= 配置区域 =================
# 1. 数据集根目录
data_root = "/root/autodl-tmp/data/nuscenes"

# 2. 源 PKL 路径 (请确保路径正确)
source_pkl = "/root/autodl-tmp/data/nuscenes/nuscenes_infos_10sweeps_val.pkl"

# 3. 目标保存路径
target_pkl = "/root/autodl-tmp/data/nuscenes/nuscenes_infos_10sweeps_val_night.pkl"
# ===========================================

def filter_night_scenes():
    print(f"正在加载 NuScenes 元数据: {data_root} ...")
    try:
        # 必须加载元数据才能查询场景信息
        nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=True)
    except Exception as e:
        print(f"加载失败: {e}")
        return

    print(f"正在加载源 PKL: {source_pkl} ...")
    with open(source_pkl, 'rb') as f:
        infos = pickle.load(f)
    
    print(f"原始验证集总帧数: {len(infos)}")

    night_infos = []
    
    # 打印第一个数据看看长什么样（调试用）
    if len(infos) > 0:
        print("样本数据示例 Keys:", infos[0].keys())

    print("开始筛选...")
    for i, info in enumerate(infos):
        # --- 核心修改部分 ---
        
        # 1. 获取当前帧的 Sample Token
        sample_token = info['token']
        
        # 2. 通过 API 查询 Sample 详细信息
        sample_record = nusc.get('sample', sample_token)
        
        # 3. 获取 Scene Token
        scene_token = sample_record['scene_token']
        
        # 4. 获取 Scene 描述
        scene_record = nusc.get('scene', scene_token)
        desc = scene_record['description'].lower()
        
        # ------------------
        
        # 筛选逻辑：描述中包含 night
        if 'night' in desc:
            night_infos.append(info)
            
        if i % 1000 == 0:
            print(f"已处理 {i}/{len(infos)} 帧...")

    print(f"筛选完成！")
    print(f"原始帧数: {len(infos)}")
    print(f"夜晚帧数: {len(night_infos)}")
    
    if len(night_infos) == 0:
        print("警告：结果为空！请检查描述匹配逻辑。")
        return

    # 保存
    with open(target_pkl, 'wb') as f:
        pickle.dump(night_infos, f)
    
    print(f"成功保存到: {target_pkl}")

if __name__ == '__main__':
    filter_night_scenes()