---
title: Serverless安装第三方package并构建layer
date: 2024-12-09 10:18:38
tags: 
    - serverless
categories: 肥用云计算
description: "本章就重点是 如何在 serverless 中安装第三方的 packag 和 如何去构建 layer 使得 package 可以复用。"
cover: https://qiniu.oldzhangtech.com/cover/%E5%A4%A7%E8%B0%B7%E7%BF%94%E5%B9%B3%202.jpg
---

## 前言
前面介绍的都没有进行额外的 package 安装的过程，所以本章就重点是 如何在 serverless 中安装第三方的 packag 和 如何去构建 layer 使得 package 可以复用。



> 下面是后续所有例子需求：
>
> 1. 安装 emoji package，且运行
>

## 安装方式：package 安装在本地
```shell
s init start-fc3-python -d demo1
cd demo1
 
# 安装依赖
touch code/requirements.txt
echo "emoji==2.0.0" > code/requirements.txt
pip3 install -r code/requirements.txt -t ./code

```

emoji package 就会安装在 local 的 code 文件夹中



修改 `code/index.py`

```python
# @code/index.py
from emoji import emojize

def handler(event, context):
    return emojize(":thumbs_up:")

```



部署代码

```shell
# 本地测试
s local invoke

# 部署到远端
s deploy
```



当然部署也是可以上传代码的方式，如下图，其效果是一样的。

![](https://qiniu.oldzhangtech.com/mdpic/7d8a2d84-885d-41b9-9176-1e453fbbf978_0a9076aa-14b3-429c-b353-b8a17a50d60c.png)



_**结果：**_

![](https://qiniu.oldzhangtech.com/mdpic/bef8d5d1-00c5-41fe-9602-c53a43c0087b_83dd9e42-6950-4625-bc3c-20b18fef455d.png)

![](https://qiniu.oldzhangtech.com/mdpic/d8ef8640-9b24-4ea9-9f98-018a77c1a333_662dc12b-8977-4705-97a5-ec7bcc63533f.png)

结果满足预期，但是他会把 emoji package 和 index.py 一起上传了。所以从代码大小中得知，整体会比较大。



## 安装方式：构建 layer
```shell
s init start-fc3-python -d demo2
cd demo2
 
# 安装依赖
touch code/requirements.txt
echo "emoji==2.0.0" > code/requirements.txt

 

```

修改 code/index.py

```python
# @code/index.py
from emoji import emojize

def handler(event, context):
    return emojize(":thumbs_up:")

```



关键代码来了，**构建 layer，并且上传 layer。**

```shell
s build --publish-layer

[2024-12-09 18:00:12][INFO][hello_world] You need to add a new configuration env configuration dependency in yaml to take effect. The configuration is as follows:
environmentVariables:
  PYTHONPATH: /opt/python
  
layers:
  - acs:fc:cn-shenzhen:1719759326012690:layers/demo2-layer/versions/1

```

构建成功后，提示说在 s.yaml 中添加上 environmentVariables 和 layers 节点。



💡 构建 layer 的意思就是把 requirements package 都构建一层 image layer 且上传到云端，后续的代码就是基于这层，不用另外的安装代码。如果有其他的项目想复用这些 package，直接改 s.yaml 的 layer 节点就可以了。



⚠️ 记得要把 requirements.txt 的文件要放在 和 index.py 的同级目录下。



所以往 s.yaml 里面写入上述的信息。

```yaml
props:
      ...
      environmentVariables:
        PYTHONPATH: /opt/python
      layers:
        - acs:fc:cn-shenzhen:1719759326012690:layers/demo2-layer/versions/1
```



部署代码

```shell
# 本地测试
s local invoke

# 部署到远端
s deploy
```

_****_

_**结果：**_

![](https://qiniu.oldzhangtech.com/mdpic/3182e155-8a24-4d6e-80bf-72c8890ebdf5_e3de1325-4c7c-4b4f-8ce5-cc6c9d2bb8a4.png)

因为不会上传 emoji package，代码明显是小了很多。



## 总结
1. serverless 就是 业务 和 trigger 隔离开来，本文章例子中都是完成 业务，没有 trigger；只要后续补充上 trigger，这个业务就可以串通了。
2. 部署方式，优先是构建 layer，后上传代码，因为 layer 可以复用，且减少项目的代码。
3. 如果开发/本地测试/部署，可以遵循下面的方法：

```shell
# 初始化
s init start-fc3-python -d demo2
cd demo2
 
# 安装依赖
touch code/requirements.txt
echo "emoji==2.0.0" > code/requirements.txt

# 本地测试
s local invoke

# 构建 layer
s build --publish-layer

# 部署
s deploy

```



## 资料


1. [源代码](https://gitee.com/oldmanzhang/practice_serverlsess/tree/master/p03)
2. [阿里云 serverless 说明](https://help.aliyun.com/zh/functioncompute/fc-3-0/user-guide/request-handlers?spm=a2c4g.11186623.4.2.72757bbcKfadgi&scm=20140722.H_2512964._.ID_2512964-OR_rec-V_1)
3. [Serverless Devs Docs](https://manual.serverless-devs.com/user-guide/aliyun/fc3/build/)
4. [start-fc-template](https://github.com/devsapp/start-fc?spm=5176.fcnext.0.0.18e978c8K4qMDn)







