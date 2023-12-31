## PCL 点云参数传递 与点云复制

### 参数传递

1. 参数无论是ptr 还是引用，函数内部修改，都会修改外部实参。（const PointCloud::Ptr pc，前面加const 修改数据内容也会影响外部）

2. 定义PointCloud类型传参，内部修改不会影响外部。（或const PointCloud& pc，也不影响外部，若修改会报错）

### 点云复制

1. 等号复制：指针和指针之间，无论是指针本身用等号，还是指针内容用等号，本质是同一块数据；PointCloud和PointCloud也是如此。但用指针数据赋给非指针，或反过来，则使用两块内存。
2. pcl提供了copyPointCloud函数用于点云复制：用指针的数据时，使用相同内存；其他情况，指针给变量，或变量给变量，变量给指针，都是两块不同的区域。

## 滤波算法

1. 统计滤波

2. 半径滤

3. 条件滤波

4. 高斯滤波：基于高斯核的卷积滤波实现 高斯滤波相当于一个具有平滑性能的低通滤波器。

   ```cpp
   #include <pcl/search/kdtree.h>
   #include <pcl/filters/convolution_3d.h>
   
   //-----------基于高斯核函数的卷积滤波实现------------------------
   pcl::filters::GaussianKernel<pcl::PointXYZ, pcl::PointXYZ> kernel;
   kernel.setSigma(4);//高斯函数的标准方差，决定函数的宽度
   kernel.setThresholdRelativeToSigma(4);//设置相对Sigma参数的距离阈值
   kernel.setThreshold(0.05);//设置距离阈值，若点间距离大于阈值则不予考虑
   cout << "Kernel made" << endl;
   
   pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
   tree->setInputCloud(cloud);
   cout << "KdTree made" << endl;
   
   //---------设置Convolution 相关参数---------------------------
   pcl::filters::Convolution3D<pcl::PointXYZ, pcl::PointXYZ, pcl::filters::GaussianKernel<pcl::PointXYZ, pcl::PointXYZ>> convolution;
   convolution.setKernel(kernel);//设置卷积核
   convolution.setInputCloud(cloud);
   convolution.setNumberOfThreads(8);
   convolution.setSearchMethod(tree);
   convolution.setRadiusSearch(0.01);
   
   cout << "Convolution Start" << endl;
   convolution.convolve(*cloud_filtered);
   ```

5. 双边滤波：是对带有强度值字段的点云进行处理。其主要目的是平滑点云数据以去除噪声，同时保留点云数据的边缘和细节信息。双边滤波器的基本思想是将每个点周围的点用高斯函数和距离函数进行加权平均，从而达到平滑点云数据的效果。

   ```cpp
   #include <pcl/filters/bilateral.h>  
   #include <pcl/search/flann_search.h>  
   #include <pcl/search/kdtree.h>  
   pcl::search::KdTree<PointT>::Ptr tree1(new pcl::search::KdTree<PointT>);  
   pcl::BilateralFilter<PointT> bf;  
   bf.setInputCloud(cloud);  
   bf.setSearchMethod(tree1);  
   bf.setHalfSize(sigma_s);  
   bf.setStdDev(sigma_r);  
   bf.filter(cloud_filtered);  
   ```

6. 中值滤波

7. 直通滤波

8. 空间裁剪滤波：根据输入的参数，裁剪指定区域

   ```cpp
   #include <pcl/filters/plane_clipper3D.h>
   pcl::PointCloud<PointT>::Ptr plane_clip(const pcl::PointCloud<PointT>::Ptr& src_cloud, const Eigen::Vector4f& plane, bool negative) 
   {
          pcl::PlaneClipper3D<PointT> clipper(plane);
          pcl::PointIndices::Ptr indices(new pcl::PointIndices);
          clipper.clipPointCloud3D(*src_cloud, indices->indices);
          pcl::PointCloud<PointT>::Ptr dst_cloud(new pcl::PointCloud<PointT>);
          pcl::ExtractIndices<PointT> extract;
          extract.setInputCloud(src_cloud);
          extract.setIndices(indices);
          extract.setNegative(negative);
          extract.filter(*dst_cloud);
          return dst_cloud;
   }
   cloud_out = plane_clip(cloud, Eigen::Vector4f(1.0,1.0, 1.0,1.0), false);
   ```

9. 移动最小二乘法光滑滤波

   ```cpp
   #include <pcl/surface/mls.h>
   pcl::MovingLeastSquares<PointT, PointT> mls;  // 定义最小二乘实现的对象mls
   mls.setComputeNormals (false);  //设置在最小二乘计算中是否需要存储计算的法线
   mls.setInputCloud (cloud_filtered);        //设置待处理点云
   mls.setPolynomialOrder(2);             // 拟合2阶多项式拟合
   mls.setPolynomialFit (false);  // 设置为false可以 加速 smooth
   mls.setSearchMethod (treeSampling);    // 设置KD-Tree作为搜索方法
   mls.setSearchRadius (0.05); // 单位m.设置用于拟合的K近邻半径
   mls.process (*cloud_smoothed);        //输出
   ```

## 配准

### [NDT粗配准](https://www.cnblogs.com/21207-ihome/p/8039741.html)

配准原理：先求原点云划分的每个网格的点云的概率密度分布，然后计算待配准点云旋转后的概率，计算最大概率的旋转矩阵输出。

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
   
     
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   

### 4PCS

在ransac的基础上，使用共面的4个点进行配准。该方法适用于重叠区域较小或者重叠区域发生较大变化场景点云配准，无需对输入数据进行预滤波和去噪，算法能够快速准确的完成点云配准。

### K4PCS

1. 体素下采样，然后3D提取关键点
2. 基于关键点而不是全局点云进行匹配，加快了匹配速度

### SAC-IA

1. 随机在原点云采样m个点
2. 然后在目标点云找寻FPFH特征相似的m个点。
3. 利用这m组点计算变换矩阵
4. 保存变换误差
5. 迭代找寻最小误差

### [ICP精配准](https://zhuanlan.zhihu.com/p/397926700)

1. 找最近点
2. 根据对应点关系，求svd，得到变换关系R
3. 调整对应点，最小化loss，求解变换关系
4. 迭代计算，直到收敛

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


## 计算PCA的两种方式
1. Eigen 求解
```C++
Eigen::Vector4f pcaCentroidtarget;//容量为4的列向量
pcl::compute3DCentroid(*target, pcaCentroidtarget);//计算目标点云质心
Eigen::Matrix3f covariance;//创建一个3行3列的矩阵，里面每个元素均为float类型
pcl::computeCovarianceMatrixNormalized(*target, pcaCentroidtarget, covariance);//计算目标点云协方差矩阵
Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);//构造一个计算特定矩阵的类对象
Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();//eigenvectors计算特征向量
Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();//eigenvalues计算特征值
cout << eigenVectorsPCA <<endl;
```
输出如下：

-0.0274781  -0.0814317  0.9963

0.0344833   -0.99616 -0.0804692

0.999027 0.0321446   0.0301806

第一列为最小特征向量。
2. 直接pca
```c++
#include <pcl/common/pca.h>
pcl::PCA<pcl::PointXYZ> pca;       // PCA算法
pcl::PointCloud<pcl::PointXYZ> objProj;
pca.setInputCloud(target);                          //设置输入点云
Eigen::Matrix3f EigenSpaceObj = pca.getEigenVectors(); //获取特征向量
cout<<EigenSpaceObj<<endl;
```
输出如下：

0.9963   -0.0814287  -0.0274762

-0.0804664  -0.996161   0.0344773

 0.0301782  0.032139 0.999028

第一列为最大特征向量。
3. 求两个特征向量的变换矩阵
```c++
//求两向量旋转矩阵
Eigen::Matrix3d rotationMatrix;
Eigen::Vector3d v1(eigenVectorsPCA(0,2), eigenVectorsPCA(1,2), eigenVectorsPCA(2,2));
Eigen::Vector3d w1(model_eigenVectorsPCA(0,2),model_eigenVectorsPCA(1,2),model_eigenVectorsPCA(2,2) );
rotationMatrix = Eigen::Quaterniond::FromTwoVectors(w1, v1).toRotationMatrix();

std::cout << "rotationMatrix  is:" << std::endl;
cout << rotationMatrix <<endl;
cout << rotationMatrix * w1 <<endl;

Eigen::Matrix4d transformMatrix;
transformMatrix.setIdentity();
// 平移向量
Eigen::Vector3d t(pcaCentroidsource[0] - pcaCentroidtarget[0], pcaCentroidsource[1] - pcaCentroidtarget[1],
pcaCentroidsource[2] - pcaCentroidtarget[2]);
transformMatrix.block<3, 3>(0, 0) = rotationMatrix;
transformMatrix.topRightCorner(3, 1) = -t;
```

## PCL拟合圆柱

1. ransac 随机采样一致性原理
   + 先随机选样本点集作为内点集
   + 从内点集选择最小样本，拟合相关参数
   + 计算其他点到拟合圆柱的距离，根据阈值确定内点集
   + 如果当前内点集大于之前最大内点集，则更新
   + 重复2-4，直到达到预定迭代参数
   + 使用最大内点集的参数作为圆柱拟合参数
   
2. 拟合代码
	```c++
	pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
	
	seg.setOptimizeCoefficients(true);        //设置对估计的模型系数需要进行优化
	seg.setModelType(pcl::SACMODEL_CYLINDER); //设置分割模型为圆柱型
	seg.setMethodType(pcl::SAC_RANSAC);       //设置采用RANSAC作为算法的参数估计方法
	seg.setNormalDistanceWeight(0.1);         //设置表面法线权重系数
	seg.setMaxIterations(5000);               //设置迭代的最大次数
	seg.setDistanceThreshold(0.05);           //设置内点到模型的距离允许最大值 
	seg.setRadiusLimits(0, 0.1);              //设置估计出圆柱模型的半径范围
	seg.setInputCloud(cloud_filtered2);
	seg.setInputNormals(normals2);
	//获取圆柱模型系数和圆柱上的点
	seg.segment(*inliers_cylinder, *coefficients_cylinder);
	```
	模型系数(point_on_axis.x, point_on_axis.y, point_on_axis.z, axis_direction.x, axis_direction.y, axis_direction.z,radius). 前三个系数表示圆柱体轴心上的一点, 后三个系数表示沿着轴的向量，最后一个系数表示半径.

## pcl 基本操作

1. 计算法线 pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::PointXYZRGBNormal> 	normalEstimation;
2. 计算两向量的夹角 pcl::getAngle3D(v1, cur_normal);
3. 滤波
   + 直通滤波 pcl::PassThrough<pcl::PointXYZ> pass; 
   + 体素滤波 pcl::VoxelGrid<pcl::PointXYZ> sor;

## 点云拟合分割

### ransac拟合分割

### 最小二乘拟合

### 其他几何分割

1. 欧式聚类：根据点云距离进行分类

   ```c++
   #include<pcl/segmentation/extract_clusters.h> 
   // --------------桌子平面上的点云团,　使用欧式聚类的算法对点云聚类分割----------
   pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
   tree->setInputCloud(cloud_filtered);              // 桌子平面上其他的点云
   vector<pcl::PointIndices> cluster_indices;        // 点云团索引
   pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;// 欧式聚类对象
   ec.setClusterTolerance(0.02);                     // 设置近邻搜索的搜索半径为2cm（也即两个不同聚类团点之间的最小欧氏距离）
   ec.setMinClusterSize(100);                        // 设置一个聚类需要的最少的点数目为100
   ec.setMaxClusterSize(25000);                      // 设置一个聚类需要的最大点数目为25000
   ec.setSearchMethod(tree);                         // 设置点云的搜索机制
   ec.setInputCloud(cloud_filtered);
   ec.extract(cluster_indices);                      // 从点云中提取聚类，并将点云索引保存在cluster_indices中
   ```

2. 区域生成分割

   + 从曲率最小的作为种子点开始

   + 从邻域寻找法向相差小的加入聚类C,曲率相差小的加入种子点集合Q

   + 当前种子点查找完毕后，删除，重复Q的种子点查找，直到Q为空，完成一个区域生成

   + 剩余点曲率排序，从小的开始作为种子点，重复以上步骤

    ```c++
   #include<pcl/segmentation/region_growing.h> 
   //区域生长聚类分割对象 <点，法线>
    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(50);                         // 最小的聚类的点数
    reg.setMaxClusterSize(1000000);                    // 最大的聚类的点数
    reg.setSearchMethod(tree);                         // 搜索方式
    reg.setNumberOfNeighbours(30);                     // 设置搜索的邻域点的个数
    reg.setInputCloud(cloud);                          // 输入点云
    if (Bool_Cuting)reg.setIndices(indices);           // 通过输入参数设置，确定是否输入点云索引
    reg.setInputNormals(normals);                      // 输入的法线
    reg.setSmoothnessThreshold(SmoothnessThreshold / 180.0 * M_PI);// 设置平滑阈值，法线差值阈值
    reg.setCurvatureThreshold(CurvatureThreshold);     // 设置曲率的阈值
   
    std::vector <pcl::PointIndices> clusters;
    reg.extract(clusters);                             // 获取聚类的结果，分割结果保存在点云索引的向量中
    ```

 3. 基于法线微分的分割

    DoN算法应该算是一种比较先进的点云滤波算法。分割本质上还是由欧式分割算法完成的。算法的目的是在去除点云低频滤波，低频信息（例如建筑物墙面，地面）往往会对分割产生干扰，高频信息（例如建筑物窗框，路面障碍锥）往往尺度上很小，直接采用 基于临近信息 的滤波器会将此类信息合并至墙面或路面中。所以DoN算法利用了多尺度空间的思想。

    + 小尺度上计算点云法线
    + 大尺度上计算点云法线
    + 滤除差异小的点
    + 欧式分割

 4. 超体素分割

    + 对均匀分布的种子点生长形成超体素，删除没有足够的邻域体素的种子

    + 从种子聚类中心向外扩展
    + 扩展时将距离最近的体素加进来，距离的计算由颜色，法线，欧氏距离三部分，可配置权重
    + 扩展直到没有体素可加进来

 5. 渐进式形态学滤波分割

    + 形态学滤波先腐蚀后膨胀，滤除指定大小的窗口的点云，保留大型地物
    + 在此基础上提出渐进式（不断增大窗口的迭代运算对非地面点加以滤除）形态学滤波
    + 增加滤波器的窗口大小和高差阈值将建筑、汽车和植被等非地面物体与地面进行分割。

## DBSCAN 

1. 两个算法参数：邻域半径R和最少点数目MinPoints。这两个算法参数实际可以刻画什么叫密集：当邻域半径R内的点的个数大于最少点数目MinPoints时，就是密集。

2. 三种点：核心点（满足密集条件），边界点（不满足密集条件但在核心点的邻域半径内），噪声点（既非核心又非边界的点）

3. 四种关系：密度直达、密度可达、密度相连，非密度相连

   + 直达：若P为核心点，Q在P的邻域半径，则P到Q密度直达。（Q到P不一定直达，因为Q不一定是核心点，直达关系不具有对称性）
   + 可达：若存在核心点P1，P2,P3，若P1到P2直达，P2到P3直达，P3到Q直达，则P1到Q可达(Q到P1不一定可达，因为Q不一定是核心点，可达关系也不具有对称性）
   + 相连：若存在核心点S,S到P和Q都可达，那么P和Q密度相连，此关系具有对称性，密度相连的点则归为一类。
   + 不相连：如果两个点不属于密度相连关系，则两个点非密度相连。非密度相连的两个点属于不同的聚类簇，或者其中存在噪声点。

4. 计算步骤：DBSCAN的算法步骤分成两步。

      1. 寻找核心点形成临时聚类簇。

      扫描全部样本点，如果某个样本点R半径范围内点数目>=MinPoints，则将其纳入核心点列表，并将其密度直达的点形成对应的临时聚类簇。

      2. 合并临时聚类簇得到聚类簇。

      对于每一个临时聚类簇，检查其中的点是否为核心点，如果是，将该点对应的临时聚类簇和当前临时聚类簇合并，得到新的临时聚类簇。重复此操作，直到当前临时聚类簇中的每一个点要么不在核心点列表，要么其密度直达的点都已经在该临时聚类簇，该临时聚类簇升级成为聚类簇。

      继续对剩余的临时聚类簇进行相同的合并操作，直到全部临时聚类簇被处理。

5. DBSCAN算法具有以下特点：

      优点：
      
      - 基于密度，对远离密度核心的噪声点鲁棒
      
      - 无需知道聚类簇的数量
      
      - 可以发现任意形状的聚类簇
      
      
      缺点：
      
      + 当数据量增大时，要求较大的内存支持I/O消耗也很大；
      
      + 当空间聚类的密度不均匀、聚类间距差相差很大时，聚类质量较差，因为这种情况下参数MinPts和Eps选取困难。
      
      + 算法聚类效果依赖与距离公式选取，实际应用中常用欧式距离，对于高维数据，存在“维数灾难”。

## 点云绕任意轴旋转

### Eigen::Matrix4f方法

1. 绕Z轴旋转
    ```c++
    Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
    // 定义一个旋转矩阵 (见 https://en.wikipedia.org/wiki/Rotation_matrix)
    float theta = M_PI/4; // 弧度角
    transform_1 (0,0) = cos (theta);
    transform_1 (0,1) = -sin(theta);
    transform_1 (1,0) = sin (theta);
    transform_1 (1,1) = cos (theta);
    ```

### Eigen::Affine3f

```c++
Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();  
// 在 X 轴上定义一个 2.5 米的平移.
transform_2.translation() << 2.5, 0.0, 0.0;
// 和前面一样的旋转; Z 轴上旋转 theta 弧度
transform_2.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitZ()));
```

### AngleAxis(angle, axis)

绕该轴逆时针旋转angle(弧度)角度。
```c++
Eigen::AngleAxisd rotation_vector(M_PI / 4, Eigen::Vector3d(0, 0, 1)); //沿 Z 轴旋转 45 度
Eigen::Vector3d v(1, 0, 0);
Eigen::Vector3d v_rotated = rotation_vector * v;
```

### 四元素旋转

1. 基本运算

```c++
Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
cout << "quaternion = \n" << q.coeffs() << endl;   // 请注意coeffs的顺序是(x,y,z,w),w为实部，
q = Eigen::Quaterniond(rotation_matrix);
cout << "quaternion = \n" << q.coeffs() << endl;
// 使用四元数旋转一个向量，使用重载的乘法即可
v_rotated = q * v; 
```

2. 赋值

   +  // 1、直接赋值，赋值顺序为[w,x,y,z]
          Eigen::Quaterniond q1(1.0, 0.0, 0.0, 0.0);
   + // 旋转向量赋值
         // 旋转向量使用 AngleAxis, 它底层不直接是Matrix，但运算可以当作矩阵（因为重载了运算符）
         Eigen::AngleAxisd rotation_vector(M_PI / 4, Eigen::Vector3d(0, 0, 1));     //沿 Z 轴旋转 45 度
         Eigen::Quaterniond q4;
         q4= Eigen::Quaterniond(rotation_vector);
         cout << "旋转向量赋值：quaternion = \n" << q4.coeffs() << endl;   
   + // 初始化欧拉角(Z-Y-X，即RPY, 先绕x轴roll,再绕y轴pitch,最后绕z轴yaw)
         Eigen::Vector3d ea(0.785398, -0, 0);
         Eigen::Quaterniond quaternion3;
         quaternion3 = Eigen::AngleAxisd(ea[0], Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(ea[1], Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(ea[2], Eigen::Vector3d::UnitX());

3. 转换

   + 转旋转矩阵 

     // 初始化欧拉角(Z-Y-X，即RPY, 先绕x轴roll,再绕y轴pitch,最后绕z轴yaw)
         Eigen::Vector3d ea(0.785398, -0, 0);
         Eigen::Quaterniond quaternion3;
         quaternion3 = Eigen::AngleAxisd(ea[0], Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(ea[1], Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(ea[2], Eigen::Vector3d::UnitX());

   + 转旋转向量

     Eigen::Quaterniond q9(1.0, 0.0, 0.0, 0.0);
         Eigen::AngleAxisd rotation_vector9(q9);
         //或者
         Eigen::AngleAxisd rotation_vector9_1;
         rotation_vector9_1 = q9;
         cout << "rotation_vector9 " << "angle is: " << rotation_vector9.angle() * (180 / M_PI)
             << " axis is: " << rotation_vector9.axis().transpose() << endl;

   + 转欧拉角

      Eigen::Vector3d eulerAngle = q9.matrix().eulerAngles(2, 1, 0);
         cout << "yaw(z) pitch(y) roll(x) = " << eulerAngle.transpose() << endl;

## 点云点的特征属性

### PCA

通过计算法点云pca得到三个特征向量，特征值最小的对应平面的法向量。PCL中法向量计算也是利用该方法

### 曲率

点云的曲率，对应的是曲面曲率，可分为主曲率、平均曲率、高斯曲率。

1. 主曲率：某点的无数个曲线中存在最大和最小的两个曲线曲率（可证明是相互垂直的），这里的最大最小曲率既是主曲率。
2. 平均曲率：主曲率的最大最小曲率的平均值即为平均曲率
3. 高斯曲率：主曲率的最大最小曲率的乘积。



## PCL 边界提取

### 若边界法线变化大

```c++
oundEst.setInputCloud(cloud);
boundEst.setInputNormals(normals);
boundEst.setRadiusSearch(0.02);
boundEst.setAngleThreshold(M_PI / 2);//边界判断时的角度阈值
```

### 平面边界提取

Alpha Shapes算法思想如下：

1. 根据邻域半径R,选择任意一点P搜索2*R内的点集Q
2. 任选一点p1,构建以p,p1为圆上点的两个圆，半径为R.
3. 除去p1,若剩余Q内点到某个圆心的距离均大于R,则p为边界点
4. 若不满足条件3，则换一个p1点计算，若均不满足条件2,3；则p不是边界点.

```c++
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
pcl::ConcaveHull<pcl::PointXYZ> chull;
chull.setInputCloud(cloud); // 输入点云为投影后的点云
chull.setAlpha(0.1);        // 设置alpha值为0.1
chull.reconstruct(*cloud_hull); 
```

### 边界点按顺序提取

两种方法计算。

1. 按向量叉乘，若向量a,b叉乘，结果小于0，则b在a的顺时针方向，若结果大于0，则b在a的逆时针方向。
2. 计算点与质心的角度，然后排序。



## PCL 关键点提取

1. ISS关键点提取

   + 对于一点P，选择邻域半径内的点
   + 计算权重系数
   + 根据权重系数，计算半径内的点Q到P两点间的协方差矩阵
   + 计算协方差矩阵特征值，按从大到小排列
   + 设置两个阈值，并满足特征值之比小于两个阈值的点即为特征点
   + 对半径内R的点进行NMS操作，保留主方向变化较大的点

2. Harris关键点提取

   是由2d图像上的算法演化而来，利用窗口移动，判断灰度是否发生较大变化，若发生较大变化，则认为存在角点。3D算法不是判断灰度而是判断法线

   + 根据邻域的点数，计算该邻域的法线的协方差矩阵M。
   + 根据角点的响应函数，计算响应值：R = det(M) - 0.004*(tr(M) * tr(M))
   + 根据响应阈值判断当前点是否是角点。

3. sift关键点检测原理

   2d图像的sift检测可参考[图像sift原理](https://blog.csdn.net/qq_40369926/article/details/88597406)

   + 高斯模糊
   + 多尺度金字塔
   + 图像差分
   + 比较差分后的相邻8个点 + 多尺度上下 9*2 共计26个点。当前点为最大或最小则为特征点。

   算法特点：具有较好的旋转，尺度不变性，能捕捉到良好的局部特征。适合物体识别，位姿估计等高精度场景。

   3Dsift有三种变换：

   + Z方向约束，不考虑Z值，与2D相同
   + 基于曲率的不变特征
   + 基于颜色的特征约束

4. NARF关键点

   算法特点：计算效率高，适合处理大规模点云，如三维重建，slam等应用场景。
   
   NARF(Normal Aligned Radial Feature)关键点是为了从==深度图像==中提取物体提出的。NARF关键点提取要求：提取过程必须将边缘以及物体表面变化信息考虑在内；关键点位置必须稳定，可以在不同视角时被重复探测；关键点所在位置必须有稳定的支持区域，可以计算描述子并进行唯一的法向量估计。
   提取步骤：
   
   + 遍历每个深度图像点，通过寻找在近邻区域有深度突变的位置进行边缘检测。
   + 遍历每个深度图像点，根据近邻区域的表面变化决定一种测度表面变化的系数，以及变化的主 方向。
   + 根据第二步找到的主方向计算兴趣值，表征该方向与其他方向的不同，以及该处表面的变化情 况，即该点有多稳定。
   + 对兴趣值进行平滑过滤。
   + 进行无最大值压缩找到最终的关键点，即为 NARF 关键点。

## PCl特征描述

1. PFH（point feature histograpm)
   + 计算每个点的法线
   + 对于邻域内两个点，计算两点的法线差异特征（q1,q2,q3)
   + 对于邻域内K个点，任意两点计算特征，则共计算k*(k-1)  / 2次。
   + 将所有计算的差异特征（q1,q2,q3)划分到5 * 5 * 5 = 125个区间内，就是维度125的特征直方图
   
2. FPFH
   + 只计算P与邻域内k个点的点对特征，生成SPFH
   + P点的FPFH为邻域内所有点的SPFH的特征加权得到。计算复杂度由nk * k 降为 n * k
   + 直方图的维度是33（将每个维度划分为11个区间）
   
3. VFH（view point feature histograpm)：主要用于聚类、识别
   + 基于FPFH，计算参考点为视点，关于质心的FPFH特征 45 * 3 个维度
   + 关于质心1个形状分布（质心到点的距离 / 最大距离)  45 个维度
   + 视角方向与点的法向量的角度值 128个维度
   
4. SHOT352

   基于ISS协方差计算，以及特征值及向量计算。并分为内外球，南北半球，8个时区，共2 * 2 * 8 * 11 = 352.

   + 计算邻域的协方差矩阵，特征值从大到小排列
   + 对每个邻域计算点的法向量与查询点法向量的夹角，为解决法向量的二义性，以特征向量的主方向为正方向
   + 对划分的每个区域统计角度值并分为11个bin
   + 为了弱化边缘效应，进行四线性插值，线性插值主要基于距离进行线性分配
   
5. 旋转图像

   + 对某一点P，其法向量为轴，绕该轴旋转
   + 将扫描到的点Q投影到法平面上，投影点到P点的距离为alpha，投影点到Q的距离为beta
   + 统计（alpha,beta)的点数，同时对应到二维图像上，便形成旋转特征图像

6. 3DSC

   + 以点P为中心，建立多个不同半径的同心球
   + 统计落在不同区域的点的个数，每个点有权重（与该区域的体积，密度相关）



## 深度学习分类方法

1. pointnet
2. pointnet++
3. GCN算法

## 深度学习检测方法

1. voxelnet
2. pointPilars
3. pointRCNN

## 深度学习特征描述方法

1. USIP

   网络设计基于两个假设：

   + 同一物体的关键点无论如何旋转平移变化，关键点都不会变化；

   + 关键点与尺度有关，在小尺度下某个点是特征点，那么在大尺度下可能不是。

   主要步骤如下：

   + 原始点云 X 输入进一个Feature Proposal NetWork网络中，输出一些关键点 A ；

   + 将原始点云随机旋转变化一个姿态 T ，新生成的点云 X2 同样输入Feature Proposal NetWork网络中，输出一些关键点 B；

   + 将关键点 B变化姿态 T ，理论上如果检测到相同的关键点，那么变化后的关键点 B 与 A 应该重合，因此根据此，设计损失函数进行训练。

2. so-net

## 学习资料参考

1. PCL(Point Cloud Library)学习指南&资料推荐（2023版）

   https://zhuanlan.zhihu.com/p/268524083 
2. PCL点云处理算法汇总
	https://zhuanlan.zhihu.com/p/457487902

