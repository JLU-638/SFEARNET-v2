#!/bin/bash
#SBATCH --job-name=job_zh           # 作业名称
#SBATCH --partition=gpujl,L40     # 指定作业所属的分区
#SBATCH --nodes=1                 # 指定作业所需的节点数
##SBATCH --exclude=node14         # 排除有问题的节点
##SBATCH --nodelist=node52  # 指定具体节点
##SBATCH --ntasks=               # 指定作业所需的节点数
#SBATCH --ntasks-per-node=1       # 指定每个节点上运行的任务数
#SBATCH --cpus-per-task=8         # 指定每个任务所需的CPU核心数
#SBATCH --mem=32GB                # 指定作业所需的内存总量
#SBATCH --gres=gpu:1              # 指定作业所需的GPU数量
#SBATCH --time=01-23:59:59        # 指定作业的最大运行时间
#SBATCH --output=./slurm-logs/jobID-%j_消融伪变化抑制模块.log       # 指定作业的标准输出文件
#SBATCH --error=./slurm-logs/jobID-%j_消融伪变化抑制模块.log       # 指定作业的标准错误输出文件。

# 设置中文环境
export LANG=zh_CN.UTF-8
# export LC_ALL=zh_CN.UTF-8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=32

# 在此处添加您要运行的命令或脚本
srun python -u train.py