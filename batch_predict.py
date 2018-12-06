#!/usr/bin/env python3
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

import os
import argparse
import time
import datetime
import cv2
import numpy as np
import uuid
import json

import functools
import logging
import collections

checkpoint_path = './east_icdar2015_resnet_v1_50_rbox'

import tensorflow as tf
import model
from icdar import restore_rectangle
import lanms
from eval import resize_image, sort_poly, detect

input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

f_score, f_geometry = model.model(input_images, is_training=False)

variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
saver = tf.train.Saver(variable_averages.variables_to_restore())

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
saver.restore(sess, model_path)

def predictor(img):
    begin = time.time()
    im_resized, (ratio_h, ratio_w) = resize_image(img)
    print('Prepare image', (time.time()-begin)*1000)
    begin = time.time()
    score, geometry = sess.run(
        [f_score, f_geometry],
        feed_dict={input_images: [im_resized[:,:,::-1]]})

    print('Forward Pass', (time.time()-begin)*1000)
    begin = time.time()
    boxes, timer = detect(score_map=score, geo_map=geometry, timer={})
    print('Detect (NMS, etc...)', (time.time()-begin)*1000)
    begin = time.time()

    if boxes is not None:
        scores = boxes[:,8].reshape(-1)
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

    text_lines = []
    if boxes is not None:
        text_lines = []
        for box, score in zip(boxes, scores):
            box = sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                continue
            tl = collections.OrderedDict(zip(
                ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                map(float, box.flatten())))
            tl['score'] = float(score)
            text_lines.append(tl)
    ret = {
        'text_lines': text_lines,
    }
    return ret


def draw_illu(illu, rst):
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
    return illu


def save_result(img, rst, original_image_name):
    session_id = 'test_001' 
    dirpath = os.path.join('static/results', session_id)
    os.makedirs(dirpath, exist_ok=True)

    new_name = original_image_name.split('.')[0]+'.png'
    # save input image
    # save illustration
    output_path = os.path.join(dirpath, new_name)
    cv2.imwrite(output_path, draw_illu(img.copy(), rst))

    # save json data
    # output_path = os.path.join(dirpath, 'result.json')
    # with open(output_path, 'w') as f:
    #     json.dump(rst, f)

    return rst



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    args = parser.parse_args()

    for f in os.listdir(args.path):
        im_path = args.path+'/'+f
        img = cv2.imread(im_path)
        rst = predictor(img)

        save_result(img, rst, f)
