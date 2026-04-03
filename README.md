SFEARNet

本仓库用于变化检测模型 SFEARNet 的训练与推理。

当前约定：
- Python 环境使用 conda 管理
- GPU 选择只在 shell 脚本里通过 CUDA_VISIBLE_DEVICES 控制
- 训练与推理代码中不再主动改写 CUDA_VISIBLE_DEVICES


一、项目结构

- data/: 数据相关代码与数据集目录
  - Dataset.py
  - dataAug.py
  - datasets/
	- LEVIR_CD_256/
- model/: 模型定义
- loss/: 损失函数
- utils/: 参数、日志、训练阶段函数
- experiment_scripts/: 实验脚本（按 模型/数据集 分层）
  - SFEARNet/LEVIR_CD_256/train_default.sh
- train.py: 训练入口
- infer_F1.py, infer_IOU.py, infer_OA.py: 推理评估入口


二、Conda 环境准备

建议使用独立环境，例如 conda_lgx。

1) 创建环境

	conda create -n conda_lgx python=3.8 -y
	conda activate conda_lgx

2) 安装 PyTorch 与 torchvision（使用 pip）

	pip install torch==2.4.1 torchvision==0.19.1

3) 安装其余依赖

	pip install tensorboardX termcolor tqdm pandas opencv-python thop


三、数据集放置

默认数据目录参数为：

	./data/datasets/LEVIR_CD_256/

目录结构需包含 train/val/test 子目录及对应 A/B/label/edge。


四、训练方式

推荐使用实验脚本（统一管理 GPU、日志、参数）：

	bash experiment_scripts/SFEARNet/LEVIR_CD_256/train_default.sh

脚本会：
- 设置 CUDA_VISIBLE_DEVICES
- 后台运行训练
- 将 stdout/stderr 重定向到 logs_of_shell 下的 training.log

如需前台直接运行：

	python train.py --num_epochs 1 --train_batchsize 2 --val_batchsize 2 --print_every_batches 50

快速联调用（只跑少量 batch）：

	MAX_TEST_BATCHES=5 bash experiment_scripts/SFEARNet/LEVIR_CD_256/train_default.sh

说明：
- MAX_TEST_BATCHES 不设置时，训练按完整数据集运行
- MAX_TEST_BATCHES=5 时，训练和验证阶段都最多只处理 5 个 batch，适合检查环境和流程


五、常用参数（train.py）

参数来自 utils/args_utils.py：
- --data_dir
- --result_dir
- --data
- --train_batchsize
- --val_batchsize
- --lr
- --num_epochs
- --gpu_id
- --model
- --lr_decline
- --weight_decay
- --lamda
- --seed
- --print_every_batches


六、日志与输出

训练过程输出目录：
- logs/model_.../
  - TF_log/: TensorBoard
  - csv/
	- train.csv: 训练 epoch 级指标
	- val.csv: 验证 epoch 级指标
	- train_batches.csv: 训练 batch 级指标
	- val_batches.csv: 验证 batch 级指标
  - best_model/
	- best_model.pth: 当前最佳模型（按 val IOU 更新）


七、推理评估

分别对应：
- infer_F1.py
- infer_IOU.py
- infer_OA.py

注意：建议与训练脚本同样在 shell 中统一设置 CUDA_VISIBLE_DEVICES。


八、常见警告与处理

1) torchvision image extension 警告

示例：
Failed to load image Python extension ... undefined symbol ...

原因通常是 torch 与 torchvision 版本不匹配。
建议保证主版本配套，例如：torch 2.4.1 对应 torchvision 0.19.1。

可执行：

	conda activate conda_lgx
	pip install torch==2.4.1 torchvision==0.19.1

2) torch.load FutureWarning

仓库已调整 backbone 权重加载为 weights_only=True，避免该警告。

3) ReduceLROnPlateau verbose 警告

仓库已去除 verbose 参数，避免该警告。


九、实验脚本规范

推荐目录层级：

	experiment_scripts/<ModelName>/<DatasetName>/

例如：

	experiment_scripts/SFEARNet/LEVIR_CD_256/train_default.sh

后续新增模型或数据集时，复制该脚本并仅修改语义变量区即可。


十、使用 byobu 跑实验

如果你在远程服务器训练，建议用 byobu 管理会话，避免断线导致任务中断。

1) 查看已有会话

	byobu list-sessions

2) 指定名字创建会话

	byobu new-session -d -s sfearnet_train

说明：
- -s 后面是会话名
- -d 表示先在后台创建，不立即进入

3) 进入自己创建过的会话

	byobu attach -t sfearnet_train

4) 会话内窗口管理

- 新建窗口：按 F2
- 切换窗口：按 F3/F4（上一窗口/下一窗口）
- 重命名当前窗口：按 F8
- 关闭当前窗口：在该窗口执行 exit（或 Ctrl + d）
- 分离会话（后台继续跑）：按 F6

等价命令（可选）：
	byobu new-window -t sfearnet_train
	byobu rename-window -t sfearnet_train:1 train

5) 推荐实践与注意事项

- 训练前先进入项目目录，再运行脚本：
	cd /home/fengruyue/lvguixin/SFEARNet
	bash experiment_scripts/SFEARNet/LEVIR_CD_256/train_default.sh
- 用日志文件跟踪训练：
	tail -f logs_of_shell/你的实验目录/training.log
- 需要快速验证流程时，可临时加环境变量限制批量：
	MAX_TEST_BATCHES=5 bash experiment_scripts/SFEARNet/LEVIR_CD_256/train_default.sh
- 只在 shell 脚本中设置 CUDA_VISIBLE_DEVICES，代码内部不改写 GPU 环境变量
- 训练任务结束后，建议手动关闭无用窗口/会话，保持 byobu 清爽

常用补充命令：
	byobu kill-session -t sfearnet_train
	byobu kill-server

