# (MGN_ReID) Multiple Granularity Network 多粒度行人重识别网络


#### Reproduction of paper: https://arxiv.org/abs/1804.01438 Learning Discriminative Features with Multiple Granularities for Person Re-Identification

#### Training Step

- Prepare data

  Donload: (Now support 4 datasets)

  - [x] Market1501 data: http://www.liangzheng.org/Project/project_reid.html

  - [ ] DukeMTMC

  - [x] Occ-Duke

  - [ ] MSMT17

  **Put the data into ==/dataset== fold**

- Modify the **config.py** 

  - add dataset directoty to --root
  - change dataset name --dataset
  - modify other parameter

- Start to train

  run **`python train.py`**

  

- Test
  - After finishing the training, modify the `--test_weight`  in **config.py**, then run  **`python test.py`**

```
@ARTICLE{2018arXiv180401438W,
    author = {{Wang}, G. and {Yuan}, Y. and {Chen}, X. and {Li}, J. and {Zhou}, X.},
    title = "{Learning Discriminative Features with Multiple Granularities for Person Re-Identification}",
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1804.01438},
    primaryClass = "cs.CV",
    keywords = {Computer Science - Computer Vision and Pattern Recognition},
    year = 2018,
    month = apr,
    adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180401438W},
    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```



-------------------------------------------------------------------------------------------------------------------------------------------

#### MGN论文复现: https://arxiv.org/abs/1804.01438 Learning Discriminative Features with Multiple Granularities for Person Re-Identification

#### 训练步骤

- 数据准备

  下载: (目前支持4个数据集，后续会持续更新)

  - [x] Market1501 data: http://www.liangzheng.org/Project/project_reid.html

  - [ ] DukeMTMC

  - [x] Occ-Duke

  - [ ] MSMT17

  将数据放入 `/dataset`文件下

- 修改 **config.py** 

  - 在 --root 中修改数据集位置
  - 更改数据集名字 --dataset （'market1501', 'occ_duke',......）
  - 更改其他参数

- 开始训练

  运行 **`python train.py`**

  

- 测试
  - 在训练完后 修改**config.py**中的`--test_weight`， 更改为得到的pth的地址, 然后运行  **`python test.py`**