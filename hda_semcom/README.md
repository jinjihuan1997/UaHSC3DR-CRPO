# HDA-SemCom

当前项目用于仿真 UAV 场景下的混合语义通信与资源分配。最新实现已经从早期的“ROI/背景分裂传输”调整为更简单的双层结构：

```text
RGB 图像             -> JSCC/语义通信传输
Depth + valid mask   -> 数字链路传输，传输前按 PNG bytes 统计负载
UAV 轨迹坐标         -> A2G 信道模型生成当前 SNR
PPO                  -> 根据信道状态和数字负载选择数字链路子载波数 k_d
```

当前不再传输 `roi_mask`，也不再把 RGB 按 ROI/非 ROI 拆开。RGB 整图统一走 JSCC，depth/mask 作为辅助数字数据传输。

## 当前系统建模

每个通信时隙对应 UAV 轨迹上的一个 step。

```text
MAGICIAN 轨迹 step
  -> UAV 坐标
  -> 当前 RGB/depth/mask 图像帧
  -> A2G 信道计算 SNR
  -> PPO 选择数字链路子载波数 k_d
  -> RGB 使用剩余子载波进行 JSCC 恢复
  -> depth/mask 根据数字链路 bit budget 得到 aux_ratio
```

资源分配关系：

```text
K_total = 24
k_d ∈ {10, 12, 14, 16, 18, 20, 22}
k_rgb = K_total - k_d
```

奖励函数：

```text
r_t = alpha * norm(PSNR_rgb) + (1 - alpha) * aux_ratio
```

其中：

```text
norm(PSNR_rgb) = clip((PSNR_rgb - PSNR_min) / (PSNR_max - PSNR_min), 0, 1)
aux_ratio = min(aux_budget_bits / aux_payload_bits, 1)
```

默认 `alpha=0.5`，表示 RGB 恢复质量和 depth/mask 数字传输比例权重相同。

## 数据输入

数据路径由 [configs/default.yaml](configs/default.yaml) 控制。当前主要读取：

```text
clean_image   RGB 目标图像
raw_depth     深度图
valid_mask    有效区域 mask
scene_id      场景名称，例如 fushimi
trajectory_id 轨迹编号
frame_id      轨迹中的帧编号
```

通信模型训练会把指定场景下的样本作为普通图像样本训练，不按轨迹顺序训练。PPO 环境才按照轨迹 step 顺序执行。

## 一键流程

新增的总流程脚本是：

```text
scripts/run_scene_pipeline.py
```

它串联四件事：

```text
1. 从 hda_semcom_dataset/scenes/<scene> 重建 manifest
2. 训练指定场景的通信模型
3. 生成 train/test 离线通信性能表
4. 从数据集内 poses/*.json 读取 UAV 位姿，并转换为米制全局坐标
```

先用 `--dry-run` 检查命令：

```bash
/home/king/miniconda3/envs/hda_semcom/bin/python scripts/run_scene_pipeline.py \
  --scene-id fushimi \
  --dry-run
```

正式运行：

```bash
/home/king/miniconda3/envs/hda_semcom/bin/python scripts/run_scene_pipeline.py \
  --scene-id fushimi
```

默认会从 `--dataset-root` 指向的数据集读取输入：

```text
/home/king/Downloads/Projects/TCOM/datasets/hda_semcom_dataset/scenes/fushimi/observations/traj*/frame_manifest.jsonl
/home/king/Downloads/Projects/TCOM/datasets/hda_semcom_dataset/scenes/fushimi/observations/traj*/poses/*.json
```

不要传 `--trajectory-input-dir`，除非你要覆盖为外部轨迹 CSV 目录。传了该参数后，坐标转换会改为读取那个外部目录，而不是数据集内的 `poses/*.json`。

默认输出：

```text
checkpoints/stage3_final.pt
outputs/offline_lookup_fushimi_slot05_k10_22_snr0_20_step2_train.csv
outputs/offline_lookup_fushimi_slot05_k10_22_snr0_20_step2_test.csv
outputs/magician_uav_trajectory/fushimi_trajectories_global_vehicle2000.csv
```

如果已经训练过通信模型，只想重新生成离线表和轨迹坐标：

```bash
/home/king/miniconda3/envs/hda_semcom/bin/python scripts/run_scene_pipeline.py \
  --scene-id fushimi \
  --skip-train
```

如果只做坐标转换：

```bash
/home/king/miniconda3/envs/hda_semcom/bin/python scripts/run_scene_pipeline.py \
  --scene-id fushimi \
  --skip-train \
  --skip-lookup
```

## 单独训练通信模型

```bash
/home/king/miniconda3/envs/hda_semcom/bin/python scripts/train.py \
  --config configs/default.yaml \
  --train-scene-id fushimi
```

训练结果保存在：

```text
checkpoints/stage1.pt
checkpoints/stage2.pt
checkpoints/stage3_final.pt
```

## 生成离线表

训练表：

```bash
/home/king/miniconda3/envs/hda_semcom/bin/python scripts/build_offline_lookup_table.py \
  --config configs/default.yaml \
  --ckpt checkpoints/stage3_final.pt \
  --split train \
  --scene-id fushimi \
  --digital-subcarriers 10 12 14 16 18 20 22 \
  --snrs 0 2 4 6 8 10 12 14 16 18 20 \
  --out outputs/offline_lookup_fushimi_slot05_k10_22_snr0_20_step2_train.csv
```

测试表：

```bash
/home/king/miniconda3/envs/hda_semcom/bin/python scripts/build_offline_lookup_table.py \
  --config configs/default.yaml \
  --ckpt checkpoints/stage3_final.pt \
  --split test \
  --scene-id fushimi \
  --digital-subcarriers 10 12 14 16 18 20 22 \
  --snrs 0 2 4 6 8 10 12 14 16 18 20 \
  --out outputs/offline_lookup_fushimi_slot05_k10_22_snr0_20_step2_test.csv
```

离线表的核心字段：

```text
sample_idx, scene_id, trajectory_id, frame_id
snr_db, k_d, k_rgb
rgb_symbols, digital_re, mcs
aux_payload_bits, aux_budget_bits, aux_ok, aux_ratio
rgb_psnr, valid_psnr, invalid_psnr
```

## 坐标转换

MAGICIAN 的原始轨迹坐标不直接用于通信信道。当前项目中使用脚本转换为米制全局坐标：

```text
scripts/convert_magician_trajectories.py
```

默认假设：

```text
1 MAGICIAN raw coordinate unit = 2 m
UAV 最低高度 = 20 m
车辆坐标 = (0, 0, 1.5) m
建筑底部中心 = (2000, 0, 0) m
```

直接从数据集内 `poses/*.json` 转换：

```bash
/home/king/miniconda3/envs/hda_semcom/bin/python scripts/convert_magician_trajectories.py \
  --dataset-root /home/king/Downloads/Projects/TCOM/datasets/hda_semcom_dataset \
  --out outputs/magician_uav_trajectory/fushimi_trajectories_global_vehicle2000.csv \
  --scene-id fushimi \
  --scene-unit-m 2 \
  --uav-min-altitude-m 20 \
  --building-center-x-m 2000 \
  --vehicle-z-m 1.5 \
  --trajectory-id-from-filename
```

`--input-dir` / `--inputs` 只用于外部轨迹 CSV；使用现有 `hda_semcom_dataset` 时不需要它们。

输出字段可直接被 A2G 信道环境使用：

```text
uav_x_global_m, uav_y_global_m, uav_z_global_m
vehicle_x_m, vehicle_y_m, vehicle_z_m
vehicle_3d_distance_m, elevation_angle_deg
```

## CRPO-PPO 资源分配训练

当前 CMDP 环境在 [src/rl/cmdp_resource_env.py](src/rl/cmdp_resource_env.py)。状态空间为：

```text
s_t = [normalized_channel_quality, normalized_depth_payload_bits,
       normalized_map_progress, view_importance, normalized_task_progress]
```

动作空间为：

```text
a_t = (k_d, beta_d)
k_d    ∈ {4, 8, 12, 16, 20}
beta_d ∈ {0.1, 0.2, ..., 0.9}
```

A2G 信道默认参数：

```text
fc = 900 MHz
bandwidth = 1 MHz
Ptx = 20 dBm = 0.1 W
noise figure = 7 dB
shadow sigma = 3 dB
small-scale fading disabled
```

重新生成 depth lookup 表：

```bash
python scripts/build_offline_lookup_table.py \
  --config configs/default.yaml \
  --ckpt checkpoints/<your_semcom_model>.pt \
  --out outputs/offline_lookup_depth_train.csv \
  --split train \
  --snrs 0 2 4 6 8 10 12 14 16 18 20 \
  --digital-subcarriers 4 8 12 16 20
```

拟合 GSFusion 代理权重：

```bash
python scripts/fit_gsfusion_surrogate.py \
  outputs/gsfusion_real_metrics.csv \
  --out outputs/gsfusion_surrogate_weights.yaml
```

训练：

```bash
python scripts/train_crpo_ppo.py \
  --table outputs/offline_lookup_depth_train.csv \
  --trajectory-csv outputs/magician_uav_trajectory/fushimi_trajectories_global_vehicle2000.csv \
  --timesteps 100000 \
  --run-name crpo_gsfusion_depth \
  --device cuda
```

评估：

```bash
python scripts/eval_crpo_ppo.py \
  --table outputs/offline_lookup_depth_test.csv \
  --checkpoint checkpoints/crpo_gsfusion_depth.pt \
  --trajectory-csv outputs/magician_uav_trajectory/fushimi_trajectories_global_vehicle2000.csv \
  --episodes 20 \
  --device cuda
```

## 文件结构

```text
configs/default.yaml                    全部超参数和路径
src/data/dataset.py                     读取 RGB/depth/valid_mask 和真实 PNG payload
src/models/hda_semcom.py                   RGB JSCC + depth/mask 数字链路整体系统
src/channel/channels.py                 AWGN + 802.11ah-like MCS 数字预算
src/channel/a2g.py                      UAV-to-ground A2G 信道模型
src/rl/cmdp_resource_env.py             CRPO-PPO CMDP 资源分配环境
src/rl/crpo_ppo.py                      两头策略 CRPO-PPO
src/rl/mapping_quality.py               GSFusion 风格代理重建质量
scripts/train.py                        通信模型训练
scripts/build_offline_lookup_table.py   生成 sample/SNR/k_d 离线表
scripts/fit_gsfusion_surrogate.py       从真实 GSFusion 指标拟合代理权重
scripts/convert_magician_trajectories.py MAGICIAN 坐标转米制全局坐标
scripts/run_scene_pipeline.py           训练 + 离线表 + 坐标转换总流程
scripts/train_crpo_ppo.py               CRPO-PPO 训练
scripts/eval_crpo_ppo.py                CRPO-PPO 评估
```
