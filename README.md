### 记录一下

打完天池那个紧接着就搞这个，毕竟参加的都是RS的同行们，组里组了个队，先是一人一个TRACK，咱是TRACK4搞点云分割，不过之前从来没搞过点云，还好官方事先给了个baseline
是pointnet++，然后咱就直接上手硬刚，刚了三星期，做出的尝试如下：

- 各种调模型参：lr/decay, BN/decay，weight(因为样本高度不均，最多能差到60倍)，球体搜索半径，结果基本没啥bird用。
- 改采样方式：大类欠采样小类过采样，结果更难trn了。
- 改数据：体素化半径改小，采样点数增多等等，提升不明显，还试了数据扩增，然而因为点云都是XYZ的，无非就是插值（这里为了方便直接加扰动），很多点插重了等于白插，但是测试集也得这么搞才有微弱提升。    
- 改结构：仿着U-Net加skip connection，仿着FCN逐步upsample或者把encoder的feature map直接upsample然后fusion，然并卵，竟然不如baseline。
- 改策略，比如用大小类，然而第一个stage就玩不下去了。
- 改loss，crossentropy+lovasz+focal，结果相当难trn。
- 换pointSIFT，确实比pointnet++强，然而效果也没强哪里去。
- 想搞个feature extractor然后砸SVM，RF等各种Classifier来fusion一下，然而9kw个点时间成本太大。
- 最后试了试分块多尺度，tmd高架桥一个都没分对，感觉玩不下去了，溜了溜了。

最后五天队里决定集中火力搞TRACK1，baseline没有用官方给的U-Net系列，用Deeplab V3+，还做了做数据清洗，分割确实还可以，方法还是老方法：五折，数据增强，TTA这种，稍微做了点后处理
，miou就能干到75，不过DSM始终上不去单独训练的detector也没卵用，也处理不了，然后就凉了。

#### 总结

1. 要合理安排时间，做人不能太贪：TRACK4没啥好说的，本来就不是这个方向就不应该就去凑这个热闹，1是就XYZI四个维度很难搞，2是pdal的函数咱毛也不懂也不会处理，3是参数太多搞不定，
就应该一开始集中火力搞TRACK1，卡本来也不多，分开搞的话前期时间和资源浪费太多了，跑路的时机还是有点晚。
2. 一定要用**多模型**！！！！
3. 一定要用把数据**都用上**：TRACK1没有用MSI，光用RGB：23说白了还是干其他的任务挤占了TRACK1的资源。
4. 回归问题转化成分类问题更容易些（赛后和前排大佬交流所得）。
5. 图像相关问题多尺度确实有效。

感谢同队队员[@vicchu](https://github.com/vicchu)、@J.Yuan、@C.Han、@Y.Dong的并肩作战

感谢队长[@YonghaoXu](https://github.com/YonghaoXu)的大力指导与支持，以及那段每晚在操场散步交谈、剖析问题本质、探求网络玄妙的时光。

#### 后续

前排一票人开小号被t，稀里糊涂成了top10…
