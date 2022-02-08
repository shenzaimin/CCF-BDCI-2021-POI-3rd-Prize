
## ***项目依赖***
### 1.安装pytorch gpu版本
```conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch```
### 2.pip安装其他依赖
```pip install -r ./requirements.txt```

## ***程序运行***
### 1.拉取镜像
```docker pull registry.cn-shanghai.aliyuncs.com/cicero0/ccf2021poi:v1```
**权重文件全部打包镜像里**
### 2.创建容器并运行[容器创建后自动执行run.sh进行预测]
```docker run --rm --gpus all -v [本地挂载prediction_result目录]:/data/prediction_result registry.cn-shanghai.aliyuncs.com/cicero0/ccf2021poi:v1```
#####[本地挂载目录]即为gitee上传的/data/prediction_result目录在宿主机上的绝对路径  用于获取复现文件
### 3.获取预测文件
[本地挂载prediction_result目录]./prediction_result中的result.json即为生成的B榜复现结果文件

## ***算法阐述***
本方案解决方案可以概括为  结合图像位置信息和文本信息的text2text生成算法
文本生成模型采用端到端的Text2Text预训练模型T5框架，
在接解码过程中，采用sampling 解码策略。
然后基于原始blockblock OCR匹配校正识别结果进行匹配校正

### 主要算法组件为：
#### 1.位置动态排序模块
根据比赛提供的block信息，结合坐标位置信息，使用相对字符面积比作为排序依据，
对文本框进行位置排序，对排序好的文本使用特殊字符进行分隔，组成排序文本信息
#### 2.text2text  文本翻译模块
用位置排序模块生成的排序文本，作为生成模型的输入，用name作为lable进行文本翻译任务的训练
采用完全匹配的评估方式作为评测标准进行模型的评估，选取开发集表现最好的模型作为检查点保存为model1。
同时，对于原始block数目大于4的样本进行数据增强，训练得到新的模型model2。
#### 3.基于块的组合匹配模块
对于原始block的识别结果进行有限制的排列组合，
将文本生成模型的结果与排列组合结果进行字符串文本模糊匹配，取得分最高的结果作为最终生成结果。
#### 4.集成
以上两种模型各取5个检查点 进行少数服从多数的 集成投票，得到最终的体交结果

## 算法架构图
![image](https://user-images.githubusercontent.com/52025915/152908982-26da4d5b-8305-44de-a3ba-9f0d902d894b.png)
