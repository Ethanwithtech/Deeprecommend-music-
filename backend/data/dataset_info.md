# 数据集信息

项目使用的大型数据集:

1. **Million Song Dataset (MSD)**
   - 文件: `msd_summary_file.h5` (301MB)
   - 位置: 项目根目录
   - 描述: 包含歌曲元数据的H5格式文件

2. **用户播放记录**
   - 文件: `train_triplets.txt` (2.8GB)
   - 位置: 项目根目录下的train_triplets.txt文件夹
   - 描述: 包含用户-歌曲-播放次数的数据

## 数据集使用

这些大型数据集文件保留在根目录，而不是复制到backend/data目录，以避免重复占用磁盘空间。
在代码中应使用相对路径引用这些数据集文件。 