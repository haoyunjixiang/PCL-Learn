## PCL 点云参数传递 与点云复制

### 参数传递

1. 参数无论是ptr 还是引用，函数内部修改，都会修改外部实参。（const PointCloud::Ptr pc，前面加const 修改数据内容也会影响外部）

2. 定义PointCloud类型传参，内部修改不会影响外部。（或const PointCloud& pc，也不影响外部，若修改会报错）

### 点云复制

1. 等号复制：指针和指针之间，无论是指针本身用等号，还是指针内容用等号，本质是同一块数据；PointCloud和PointCloud也是如此。但用指针数据赋给非指针，或反过来，则使用两块内存。
2. pcl提供了copyPointCloud函数用于点云复制：用指针的数据时，使用相同内存；其他情况，指针给变量，或变量给变量，变量给指针，都是两块不同的区域。

## 配准

### NDT粗配准

1. 体素下采样 voxel_grid.setLeafSize(0.5, 0.5, 0.5);xyz中跨度最大的维度长度除以30得到0.5，若降采样的点太少，比如都小于500了，考虑减小格子大小以增加降采样点云，不然点太少会配准失败或精度太差。

2. ndt.setStepSize(0.5); 步长设置为与降采样网格相同大小

3. ndt.setResolution(1.0); 网格分辨率Resolution，Resolution设置为步长的两倍若配准结果不正确，增大步长，网格分辨率仍为步长的两倍；若配准结果矩阵为单位矩阵，同样地增大步长与网格分辨率

4. getFitnessScore ，输出所有对应点的距离平方和的平均值，分数越小越好

       // Deal with occlusions (incomplete targets)
       // 排除距離過大的點對
       if (nn_dists[0] <= max_range) {
         // Add to the fitness score
         fitness_score += nn_dists[0];
         nr++;
       }
       
       ....
       
       if (nr > 0)
       return (fitness_score / nr);
       
     
### ICP精配准
差不多五种实现：
（1）pcl::GeneralizedIterativeClosestPoint< PointSource, PointTarget >：
广义迭代最近点算法
（2）pcl::IterativeClosestPoint< PointSource, PointTarget, Scalar >
经典迭代最近点算法
（3）pcl::IterativeClosestPointWithNormals< PointSource, PointTarget, Scalar >：
带法向的迭代最近点算法
（4）pcl::IterativeClosestPointNonLinear< PointSource, PointTarget, Scalar >
非线性优化的迭代最近点算法
（5）pcl::JointIterativeClosestPoint< PointSource, PointTarget, Scalar >
角度空间的迭代最近点算法



## 学习资料参考

1. PCL(Point Cloud Library)学习指南&资料推荐（2023版）

   https://zhuanlan.zhihu.com/p/268524083   

