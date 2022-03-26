import numpy as np
import cv2
import sys

caffe_root = '/root/caffe-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe


net_file = './plateRegress.prototxt'
caffe_model = './plateRegress.caffemodel'

net = caffe.Net(net_file, caffe_model, caffe.TEST)


model_input_h = 64
model_input_w = 128


def img_preprocess(src_img):
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (model_input_w, model_input_h))

    img = img.astype(np.float32)
    img = img * 0.003921568
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    return img


def test_image(image_path):
    src_img = cv2.imread(image_path)
    img_data = img_preprocess(src_img)

    net.blobs['blob1'].data[...] = img_data
    out = net.forward()
    output = out['conv_blob13'].reshape((-1))

    img_h, img_w = src_img.shape[:2]
    scale_h = img_h / model_input_h
    scale_w = img_w / model_input_w

    for i in range(0, 32, 2):
        point = (int(output[i]*scale_w), int(output[i+1]*scale_h))
        cv2.circle(src_img, point, 1, (0, 0, 255), 2)

    cv2.imwrite('./result_caffe.jpg', src_img)
    #cv2.imshow('image', src_img)
    #cv2.waitKey(0)


if __name__ == '__main__':
    print('This is main .... ')
    image_path = './test.jpg'
    test_image(image_path)