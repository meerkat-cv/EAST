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
import sys

import functools
import logging
import collections
from collections import defaultdict

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



def left_to_right_intersection(b1, b2):
    P = np.array(b1[4]) # left midpoint
    # calculate its normal using the midpoints
    dx = b1[6][0]-b1[4][0]
    dy = b1[6][1]-b1[4][1]
    n = np.array([-dy,dx])

    p1 = np.array(b2[1]) # top left
    p2 = np.array(b2[0]) # bottom left
    val1 = np.dot(n, p1-P)
    val2 = np.dot(n, p2-P)

    # img = np.zeros((480,640,3), dtype=np.uint8)
    # img = cv2.circle(img, (int(P[0]), int(P[1])), 8, (100,0,0), -1)
    # img = cv2.circle(img, (int(p1[0]), int(p1[1])), 5, (100,100,100), -1)
    # img = cv2.circle(img, (int(p2[0]), int(p2[1])), 5, (100,100,100), -1)
    # img = cv2.line(img, (int(n[0])+100,int(n[1])+100), (100,100), (0,0,200), 2)


    if (val1>=0 and val2>=0) or (val1<0 and val2<0):
        return False
    else:
        return True

def minBoundingRect(hull_points_2d):
    edges = np.zeros( (len(hull_points_2d)-1,2) ) # empty 2 column array
    for i in range( len(edges) ):
        edge_x = hull_points_2d[i+1,0] - hull_points_2d[i,0]
        edge_y = hull_points_2d[i+1,1] - hull_points_2d[i,1]
        edges[i] = [edge_x,edge_y]

    # Calculate edge angles   atan2(y/x)
    edge_angles = np.zeros( (len(edges)) ) # empty 1 column array
    for i in range( len(edge_angles) ):
        edge_angles[i] = np.math.atan2( edges[i,1], edges[i,0] )

    # Check for angles in 1st quadrant
    for i in range( len(edge_angles) ):
        edge_angles[i] = np.abs( edge_angles[i] % (np.math.pi/2) ) # want strictly positive answers

    # Remove duplicate angles
    edge_angles = np.unique(edge_angles)

    # Test each angle to find bounding box with smallest area
    min_bbox = (0, sys.maxsize, 0, 0, 0, 0, 0, 0) # rot_angle, area, width, height, min_x, max_x, min_y, max_y
    for i in range( len(edge_angles) ):
        # Create rotation matrix to shift points to baseline
        # R = [ cos(theta)      , cos(theta-PI/2)
        #       cos(theta+PI/2) , cos(theta)     ]
        R = np.array([ [ np.math.cos(edge_angles[i]), np.math.cos(edge_angles[i]-(np.math.pi/2)) ], [ np.math.cos(edge_angles[i]+(np.math.pi/2)), np.math.cos(edge_angles[i]) ] ])

        # Apply this rotation to convex hull points
        rot_points = np.dot(R, np.transpose(hull_points_2d) ) # 2x2 * 2xn

        # Find min/max x,y points
        min_x = np.nanmin(rot_points[0], axis=0)
        max_x = np.nanmax(rot_points[0], axis=0)
        min_y = np.nanmin(rot_points[1], axis=0)
        max_y = np.nanmax(rot_points[1], axis=0)

        # Calculate height/width/area of this bounding rectangle
        width = max_x - min_x
        height = max_y - min_y
        area = width*height

        # Store the smallest rect found first (a simple convex hull might have 2 answers with same area)
        if (area < min_bbox[1]):
            min_bbox = ( edge_angles[i], area, width, height, min_x, max_x, min_y, max_y )

    # Re-create rotation matrix for smallest rect
    angle = min_bbox[0]   
    R = np.array([ [ np.math.cos(angle), np.math.cos(angle-(np.math.pi/2)) ], [ np.math.cos(angle+(np.math.pi/2)), np.math.cos(angle) ] ])

    # Project convex hull points onto rotated frame
    proj_points = np.dot(R, np.transpose(hull_points_2d) ) # 2x2 * 2xn

    # min/max x,y points are against baseline
    min_x = min_bbox[4]
    max_x = min_bbox[5]
    min_y = min_bbox[6]
    max_y = min_bbox[7]

    # Calculate center point and project onto rotated frame
    center_x = (min_x + max_x)/2
    center_y = (min_y + max_y)/2
    center_point = np.dot( [ center_x, center_y ], R )

    # Calculate corner points and project onto rotated frame
    corner_points = np.zeros( (4,2) ) # empty 2 column array
    corner_points[0] = np.dot( [ max_x, min_y ], R )
    corner_points[1] = np.dot( [ min_x, min_y ], R )
    corner_points[2] = np.dot( [ min_x, max_y ], R )
    corner_points[3] = np.dot( [ max_x, max_y ], R )

    return (angle, min_bbox[1], min_bbox[2], min_bbox[3], center_point, corner_points) # rot_angle, area, width, height, center_point, corner_points




def merge_boxes(boxes, img = None):
    # First we need to organize the boxes. The first point must be the one on the top left,
    # the second is the top right... Also we compute the edges mid points
    organized_boxes = []
    for b in boxes:
        mid_points = []
        for i in range(4):
            i2 = (i+1)%4
            p1 = np.array(b[i])
            p2 = np.array(b[i2])
            mid_points.append((p1+p2)/2)

        bigger_side, bigger_side_idx = 0, -1
        for i in range(4):
            i2 = (i+1)%4
            side_len = np.linalg.norm(np.array(b[i])-np.array(b[i2]))
            if side_len > bigger_side:
                bigger_side = side_len
                bigger_side_idx = i

        top_idx    = bigger_side_idx
        bottom_idx = (top_idx+2)%4
        if mid_points[top_idx][1] > mid_points[bottom_idx][1]:
            aux = top_idx
            top_idx = bottom_idx
            bottom_idx = aux

        right_idx = (top_idx+1)%4
        left_idx = (right_idx+2)%4

        p1 = b[top_idx]
        p2 = b[right_idx]
        p3 = b[bottom_idx]
        p4 = b[left_idx]
        mid1 = mid_points[top_idx]
        mid2 = mid_points[right_idx]
        mid3 = mid_points[bottom_idx]
        mid4 = mid_points[left_idx]

        img = cv2.line(img, (p1[0],p1[1]), (p2[0],p2[1]), (200,0,0), 4)
        img = cv2.line(img, (p2[0],p2[1]), (p3[0],p3[1]), (0,200,0), 4)
        img = cv2.line(img, (p3[0],p3[1]), (p4[0],p4[1]), (0,0,200), 4)
        img = cv2.line(img, (p4[0],p4[1]), (p1[0],p1[1]), (200,200,200), 4)
        img = cv2.circle(img, (int(mid2[0]), int(mid2[1])), 8, (100,0,0), -1)
        img = cv2.circle(img, (int(mid4[0]), int(mid4[1])), 8, (100,100,200), -1)

        organized_boxes.append((p4,p1,p2,p3,mid4,mid1,mid2,mid3))


    # Create a map of connections, that for each organized_box will have its left
    # and the right bounding boxes. But they only appear if they must be connected, 
    # i.e. are on the same line. Otherwise they point to -1
    closest_box = {}
    connections = defaultdict(dict)
    bbs_used = defaultdict(bool)
    for idx,b in enumerate(organized_boxes):
        closest_box[idx] = (100000000, -1)
        for idx2,b2 in enumerate(organized_boxes):
            if idx == idx2: continue
            # distance between b[right] and b2[left] < threshold
            # or they are intersecting each other
            dist = np.linalg.norm(np.array(b[2+4])-np.array(b2[0+4]))
            if dist < closest_box[idx][0] or (b[2+4][0] >= b2[0+4][0] and b2[0+4][0] > b[0+4][0]):
                if left_to_right_intersection(b, b2):
                    closest_box[idx] = (dist, idx2)
                    min_box_dist = np.linalg.norm(b[1+4]-b[3+4])*2
                    if dist < min_box_dist:
                        img = cv2.line(img, (int(b[2+4][0]),int(b[2+4][1])), (int(b2[0+4][0]),int(b2[0+4][1])), (100,100,100), 3)
        connections[idx]['left'] = -1
        connections[idx]['right'] = -1
        if idx == 0:
            print('connections[0]',connections[0])

        # cv2.imshow("Boxes", img)
        # cv2.waitKey()

    for b_idx in closest_box:
        p1 = organized_boxes[b_idx][1+4]
        p2 = organized_boxes[b_idx][3+4]
        height = np.linalg.norm(p1-p2)
        min_box_dist = height*2
        if closest_box[b_idx][0] <= min_box_dist:
            b2_idx = closest_box[b_idx][1]
            connections[b_idx]['right'] = b2_idx
            connections[b2_idx]['left'] = b_idx
            
            
        bbs_used[b_idx] = False

    next_bb = 0
    if len(organized_boxes) == 0:
        next_bb = -1
    # helper function that given a box_id, it returns a list of all the boxes that must be
    # merged with him, i.e. find a line.
    def get_connections(connections, bbs_used, curr_box):
        res = [curr_box,]
        bbs_used[curr_box] = True

        next_box = connections[curr_box]['left']
        if next_box != -1 and bbs_used[next_box] == False:
            res = res+get_connections(connections, bbs_used, next_box)
        
        next_box = connections[curr_box]['right']
        if next_box != -1 and bbs_used[next_box] == False:
            res = res+get_connections(connections, bbs_used, next_box)
        return res

    final_bbs = []
    while next_bb != -1:
        all_boxes = get_connections(connections, bbs_used, next_bb)
        all_points = []
        for box_id in all_boxes:
            for i in range(4):
                all_points.append([organized_boxes[box_id][i][0], organized_boxes[box_id][i][1]])

        all_points = np.array(all_points)
        rot_angle, area, width, height, center_point, corner_points = minBoundingRect(all_points)

        final_bbs.append(corner_points)
        for i in range(4):
            x1 = int(corner_points[i][0])
            y1 = int(corner_points[i][1])
            x2 = int(corner_points[(i+1)%4][0])
            y2 = int(corner_points[(i+1)%4][1])
            img = cv2.line(img, (x1, y1), (x2, y2), (0,200,200), 12)

        next_bb = -1
        for bb_idx in bbs_used:
            if bbs_used[bb_idx] == False:
                next_bb = bb_idx
                break
       
    return final_bbs




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
    boxes = []
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')


        d = d.reshape(-1, 2)
        # cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
        boxes.append((d[0],d[1],d[2],d[3]))
        
    final_bbs = merge_boxes(boxes)
    for bb in final_bbs:
        for i in range(4):
            pt1 = (int(bb[i][0]), int(bb[i][1]))
            pt2 = (int(bb[(i+1)%4][0]), int(bb[(i+1)%4][1]))

            illu = cv2.line(illu, pt1, pt2, (200,0,0), 4)
    
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')


        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, color=(0, 0, 200), thickness=1)
    
    return illu


def save_result(img, rst, original_image_name, max_size):
    session_id = 'test_size_{}'.format(max_size) 
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
    parser.add_argument('--max_size')
    args = parser.parse_args()

    max_size = int(args.max_size)

    for f in sorted(os.listdir(args.path)):
        im_path = args.path+'/'+f
        img = cv2.imread(im_path)
        W, H = img.shape[1], img.shape[0]
        biggest_size = H if H>W else W
        scale = max_size/biggest_size
        img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        rst = predictor(img)

        save_result(img, rst, f, max_size)
