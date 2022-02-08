
## ***项目依赖***
### 1.安装pytorch gpu版本
```conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch```
### 2.pip安装其他依赖
```pip install -r ./requirements.txt```

## ***程序运行***
### 1.拉取镜像
```docker pull registry.cn-shanghai.aliyuncs.com/cicero0/ccf2021poi:v1```
### 2.创建容器并运行[容器创建后自动执行run.sh进行预测]
```docker run --rm --gpus all -v [本地挂载prediction_result目录]:/data/prediction_result registry.cn-shanghai.aliyuncs.com/cicero0/ccf2021poi:v1```
#####[本地挂载目录]即为gitee上传的/data/prediction_result目录在宿主机上的绝对路径  用于获取复现文件
### 3.获取预测文件
[本地挂载prediction_result目录]./prediction_result中的result.json即为生成的B榜复现结果文件

## ***算法阐述***
本方案解决方案可以概括为  结合图像位置信息和文本信息的text2text生成算法

### 主要算法组件为：
#### 1.位置动态排序模块
#### 2.text2text  文本翻译模块
#### 3.基于块的组合匹配模块