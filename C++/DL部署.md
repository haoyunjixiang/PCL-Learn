## YOLOV5 部署到海思3516系列芯片

1. 模型转换  pytorch->onnx->caffe->nnie
2. 后处理参考  [参考1](https://github.com/mahxn0/Hisi3559A_Yolov5)      [参考2](https://gitee.com/shopping-tang/yolo_v5_nnie)

## pytorch学习率调整

1. 等间隔衰减

2. 指定epoch衰减

3. 指数衰减

4. 余弦衰减

5. 自适应调整学习率 lr_scheduler.ReduceLROnPlateau

6. 自定义调整学习率 lr_scheduler.LambdaLR

7. 自定义warmup 学习率调整，继承基类from torch.optim.lr_scheduler import _LRScheduler，复写get_lr函数. [博客](https://blog.csdn.net/weixin_44316581/article/details/124687305)    [代码](https://github.com/ildoonet/pytorch-gradual-warmup-lr)

   ```python
   def get_lr(self):
           if self.last_epoch > self.warmup_epoch:  # 超过warmup范围，使用CosineAnnealingLR类的get_lr()
               return self.after_scheduler.get_lr()
           else:  # warmup范围，编写线性变化，也就是上图中0-10区间内的直线
               return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.)
                       for base_lr in self.base_lrs]
   ```

   

## torch.nn.conv2d参数

1. in_channel
2. out_channel
3. kernel_size
4. stride
5. padding = 1
6. dilation
7. groups
8. bias
9. padding_mode

## SSD与YOLO的区别

1. SSD在卷积后输出，YOLO是全连接后输出
2. SSD使用先验框anchor，YOLO是直接进行预测。
3. SSD在三个卷积层进行输出，YOLO在一个全连接后输出