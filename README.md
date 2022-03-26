# plateRegress

将pytorch版本的车牌回归 plateRegress转成caffe、onnx、tensorRT 版本，便于移植不同平台。

# 文件结构说明

caffe：去除维度变换层的prototxt、caffeModel、测试图像、测试结果、测试demo脚本

onnx：onnx模型、测试图像、测试结果、测试demo脚本

tensorRT：TensorRT版本模型、测试图像、测试结果、测试demo脚本、onnx模型、onnx2tensorRT脚本(tensorRT-7.2.3.4)

# 测试结果

![image](https://github.com/cqu20160901/plateRegress_caffe_onnx_tensorRT/blob/master/tensorRT/result_tensorRT.jpg)
