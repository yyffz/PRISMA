# 观测Mask通道使用说明

## 背景

对于极轨卫星数据，由于扫描方式为延轨扫描，在轨迹外侧没有有效观测。将观测mask作为输入通道可以帮助模型：

1. **明确识别有效观测区域**：模型在编码阶段就能知道哪些区域是有效的
2. **避免学习无效区域噪声**：防止模型学习到无效区域的填充值或噪声模式
3. **提升tokenizer表示质量**：更好地学习有效区域的token表示

## 使用方法

### 1. 在数据集中提供observation_mask

在`video_dataset.py`的`__getitem__`方法中，需要提供`observation_mask`：

```python
def __getitem__(self, index):
    # ... 加载视频数据 ...
    data["video"] = video
    
    # 提供观测mask（形状：[T, H, W] 或 [1, T, H, W]）
    # 1表示有效观测，0表示无效区域
    observation_mask = self._get_observation_mask(video_path, frame_ids)
    data["observation_mask"] = observation_mask
    
    return data
```

### 2. 配置启用mask通道

在实验配置文件中设置：

```python
dataloader_train=dict(
    dataset=dict(
        target_channels=19,  # 18个数据通道 + 1个mask通道
        include_mask_channel=True,  # 启用mask通道
    ),
),
model=dict(
    config=dict(
        network=dict(
            in_channels=19,   # 包括mask通道
            out_channels=19,  # 包括mask通道
        ),
    ),
),
```

### 3. 命令行使用

```bash
python train.py experiment=satellite_multi_channel \
    dataloader_train.dataset.target_channels=19 \
    dataloader_train.dataset.include_mask_channel=True \
    model.config.network.in_channels=19 \
    model.config.network.out_channels=19
```

## 注意事项

1. **通道数计算**：如果启用mask通道，`target_channels`、`in_channels`和`out_channels`都需要+1
2. **Mask格式**：mask应该是二值化的（0或1），表示无效/有效区域
3. **Mask位置**：mask通道会被添加为最后一个通道
4. **默认行为**：如果数据中没有提供`observation_mask`，系统会自动创建一个全1的mask（表示所有区域都有效）

## 与loss_mask的区别

- **observation_mask（作为输入通道）**：模型在编码时就能看到，可以学习如何处理有效/无效区域
- **loss_mask（用于损失计算）**：只在损失计算时使用，用于屏蔽无效区域的损失

两者可以同时使用，互为补充。

