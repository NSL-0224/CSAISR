# 普通平均版的方向差分解码实验

本次更新将仓库原SAT相似度加权聚合替换为普通算术平均，并接入本地方向差分实验实现。原始 RDN、KVCA、DCR、普通平均、数量补偿和四邻域采样保留，新解码器在原注意力加权结果上增加自适应方向差分。DINOv3仅用于训练辅助关系监督，测试不加载教师。

这是实验代码，不是已经证明提高 PSNR 的成果。先跑无教师消融，再跑教师版本。

## 1 修改的具体计算

原采样循环顺序为 `(y-,x-), (y-,x+), (y+,x-), (y+,x+)`。设调制后的四个值向量为 `v1,v2,v3,v4`：

```text
dx = (-v1 + v2 - v3 + v4) / 4
dy = (-v1 - v2 + v3 + v4) / 4
dc = ( v1 - v2 - v3 + v4) / 4
base = 原来的注意力加权结果
detail = ax*dx + ay*dy + ac*dc
融合特征 = base + detail
```

`dc`是交叉差分，不是一阶对角导数。这里没有对整张图额外使用Sobel，没有移动采样点，也没有按实际采样间距归一化，因此不能将上述量称为精确图像梯度。

三种差分的系数都由轻量MLP产生，输入为学生结构描述、四个实际相对偏移、LR单位的目标像素单元尺寸。系数限制为 `0.1*tanh(raw)`；输出层零初始化，初始细节分支为零。此幅度是未经数据调优的起始值。

每个差分核的系数之和为零：四个V向量相同时，细节项严格为零。这一性质只作用于特征组合，不保证最终RGB无伪影。

## 2 文件入口

| 文件 | 作用 |
|---|---|
| `mmedited/models/common/directional_detail.py` | 差分基核、系数网络和关系损失 |
| `mmedited/models/backbones/sr_backbones/ciaosr_directional_net.py` | 新解码器；原解码文件不变 |
| `mmedited/models/restorers/ciaosr_directional.py` | 训练步骤、教师调度 |
| `mmedited/models/common/dinov3_teacher.py` | 独立教师进程客户端 |
| `tools/dinov3_teacher_worker.py` | 使用官方源码和本地权重运行教师 |
| `configs/002_directional_detail.py` | 差分分支，不用大模型 |
| `configs/003_directional_dinov3.py` | 差分分支＋DINOv3 |
| `configs/004_baseline_finetune.py` | 原模型相同预算继续训练 |
| `configs/005_directional_disabled.py` | 新解码类关闭分支的基线对照 |

## 3 服务器目录

所有命令均在本工程根目录执行。SR使用原来已经能训练普通平均版的环境。

```text
工程根目录/
  external_weights/
    dinov3/
      dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
  third_party/
    dinov3/                 # Meta官方源码，不是权重
  data/
    DIV2K/HR/              # 训练图片
    Classical/Urban100/x4/GT/
    Classical/Urban100/x4/LR/
  configs/
  tools/
```

不要把DINOv3权重放进 `pretrain_model`。权重、数据和官方源码checkout均不随本项目上传。可以用 `DINOV3_WEIGHTS`、`DINOV3_REPO` 指定其他绝对路径。

新实验继承原入口 `configs/001_ciaosr_rdn_div2k_patch_kv.py`，数据路径仍在该文件设置。仓库另一个 `001_csa_isr_rdn_div2k.py` 的原有Manga109 x2配置保持不变。当前继承设置为DIV2K训练、Urban100 x4验证，训练LR块48x48、每图2304个随机查询、batch 16。显存不足可将训练batch降到4或8，但三个对照必须一致。三份微调配置默认20k步、学习率1e-5，在10k/15k减半，每1k验证；这些设置需要根据服务器实验再调整。

## 4 先准备SR初始权重

使用同一份已训练的普通平均CSASR checkpoint，生成包含新分支初始化的权重。此操作只复制旧参数、补充新分支和调度计数，不复制优化器。

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CSASR_BASELINE_CHECKPOINT=/absolute/path/to/csasr_baseline.pth
python tools/prepare_directional_checkpoint.py \
  "$CSASR_BASELINE_CHECKPOINT" external_weights/csasr_directional_init.pth
export DIRECTIONAL_INIT_CHECKPOINT="$PWD/external_weights/csasr_directional_init.pth"
```

脚本校验旧参数缺失、额外键和形状，不会默默跳过错误的主干权重。支持restorer完整state_dict或generator裸state_dict。请使用自己的可信checkpoint。新分支采用零初始化，因此不要使用原训练的 `resume_from` 来开始新实验，应使用转换权重作为 `load_from`。

## 5 第一步：基线与无教师分支

```bash
# CPU数学检查；需能找到MMEditing的原RDN源码
python tests/test_directional_cpu.py

# 相同起点/预算的原模型对照
python tools/preflight_directional.py configs/004_baseline_finetune.py
python tools/train.py configs/004_baseline_finetune.py --gpus 1 --seed 20261118

# 方向差分分支：不需要DINOv3源码、权重或新环境
python tools/preflight_directional.py configs/002_directional_detail.py
python tools/train.py configs/002_directional_detail.py --gpus 1 --seed 20261118
```

`tools/train.py`会自动恢复相同work_dir中的latest.pth；重复实验务必指定不同 `--work-dir`，否则会继续上次训练。上述两个完整实验是顺序执行示例；先检查首批数据、损失和第一个验证点，再完成20k步。

## 6 第二步：准备DINOv3教师环境

原README使用Python3.8/PyTorch1.9/MMCV1.x，这与DINOv3官方当前代码不一定兼容。不要为加载教师直接升级旧SR环境。另建环境，例如Python3.11，按DINOv3官方说明安装匹配的PyTorch及依赖：

- 官方源码与安装说明：https://github.com/facebookresearch/dinov3
- 权重与许可：https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md

以下命令用教师环境的Python执行；它会下载官方源代码，不下载权重。可先用 `--ref` 指定你验证过的官方提交；否则记录打印的commit供复现。

```bash
export DINOV3_PYTHON=/absolute/path/to/dino-env/bin/python
"$DINOV3_PYTHON" tools/prepare_dinov3.py
# 将你已有的.pth复制到 external_weights/dinov3/，然后检查真实加载
"$DINOV3_PYTHON" tools/prepare_dinov3.py --check --device cuda:0
```

成功时应打印空间特征形状与 `frozen: True`。缺少依赖时按照所选官方源码版本的说明在教师环境中安装，不要在SR环境盲目安装整份依赖。

## 7 第三步：DINOv3辅助实验

回到SR环境，保留前面的环境变量，使用与无教师版本相同的转换权重起跑：

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export DINOV3_PYTHON=/absolute/path/to/dino-env/bin/python
export DINOV3_WEIGHTS="$PWD/external_weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
export DINOV3_REPO="$PWD/third_party/dinov3"
export DIRECTIONAL_INIT_CHECKPOINT="$PWD/external_weights/csasr_directional_init.pth"
python tools/preflight_directional.py configs/003_directional_dinov3.py --check-teacher
python tools/train.py configs/003_directional_dinov3.py --gpus 1 --seed 20261118
```

教师独立进程使用本地管道接收原始RGB，不访问网络；权重不进入SR checkpoint或优化器。默认与SR使用同一GPU，也可设置 `DINOV3_DEVICE=cuda:1` 或 `cpu`。同GPU会增加显存占用；教师micro_batch=1，只减少教师激活，并不减少3亿参数本身的存储。

教师每4步执行一次，先warmup 500步、关系损失在1000步内渐增，最大权重0.01。教师读取同一LR训练块并缩放到192x192，用ImageNet mean/std，提取第12个Transformer block（0-based layer=11）的12x12空间特征；不使用CLS或storage tokens。中层选择是实验超参数，不代表已证明优于末层。

学生结构描述和教师空间特征分别池化到最多8x8，比较非对角余弦关系矩阵，不要求通道数相同。该监督通过共享结构分支间接影响方向系数，不把教师关系当作正确方向强度。LR插值不会生成真实新细节，教师也不是精确边缘标注器。

## 8 测试与对比

```bash
python tools/test.py configs/002_directional_detail.py work_dirs/002_directional_detail/latest.pth
python tools/test.py configs/003_directional_dinov3.py work_dirs/003_directional_dinov3/latest.pth
```

测试不会创建或加载教师，即使测试配置中teacher.enabled=True。修改共同基配置的 `val_scale`、`data_type` 可测试其他集合/倍率。按相同Y通道、裁边、tile与数据退化设置比较PSNR/SSIM；不要在不同配置间直接比较。最后扩展到x2/x3/x4/x6/x8/x12，并记录额外推理延迟和显存。

建议报告三组：同预算基线、仅差分分支、差分＋DINO。不要把较长训练或不同初始化的提升算成分支收益。多卡入口仍为 `bash tools/dist_train.sh CONFIG GPU_COUNT --seed 20261118`，但DDP尚未实机验证，每个rank会启动自己的教师进程，应先单卡确认显存和调度。

## 9 验证边界

本地检查覆盖：差分方向与零和、常量特征消除、零初始化/关闭分支与基线一致、真实RDN/KVCA解码器的前后向、关系监督梯度、分块查询一致、严格权重转换。见 `DIRECTIONAL_CPU_TEST_LOG.txt`。`DIRECTIONAL_TRAIN_STEP_TEST_LOG.txt`另外检查真实restorer训练方法的调度、原始RGB传递与损失，但教师使用合成特征代替；`tests/test_teacher_transport.py`检查真实子进程的二进制传输。

未验证：你的DINOv3真实权重加载、完整MMCV训练runner、CUDA/DDP、实际数据PSNR与耗时。本工程没有附带任何训练完成的新权重。教师特征仍需在你的机器验证；本次没有上传DINO权重。旧加权版checkpoint虽然参数形状可能兼容，切换普通平均后算法行为不同，不能把旧分数当作普通平均基线。

继承工程的EDSR/SwinIR历史配置存在语法问题，不是本次入口；本实验仅使用上述RDN配置。
