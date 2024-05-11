import os
import argparse
import cv2
import numpy as np
import threading

TAN67_5 = np.tan(np.deg2rad(67.5))
TAN22_5 = np.tan(np.deg2rad(22.5))
class DIRECTION:
    # direction/orientation
    # 8 4 6
    # 2 0 1
    # 7 3 5
    NAME = ["NONE","RIGHT","LEFT","UP","DOWN","DOWN_RIGHT","UP_RIGHT","DOWN_LEFT","UP_LEFT"]
    NONE = 0
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    DOWN_RIGHT = 5
    UP_RIGHT = 6
    DOWN_LEFT = 7
    UP_LEFT = 8
    @staticmethod
    def get_perpendicular_direction(direction):
        if direction == DIRECTION.RIGHT or direction == DIRECTION.LEFT:
            return DIRECTION.UP, DIRECTION.DOWN
        if direction == DIRECTION.UP or direction == DIRECTION.DOWN:
            return DIRECTION.RIGHT, DIRECTION.LEFT
        if direction == DIRECTION.DOWN_RIGHT or direction == DIRECTION.UP_LEFT:
            return DIRECTION.UP_RIGHT, DIRECTION.DOWN_LEFT
        if direction == DIRECTION.UP_RIGHT or direction == DIRECTION.DOWN_LEFT:
            return DIRECTION.DOWN_RIGHT, DIRECTION.UP_LEFT
        return DIRECTION.NONE, DIRECTION.NONE
    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path')
    parser.add_argument('--output_path',default="./OutputImages",type=str)
    return parser.parse_args()

def GausBlur(image):
    gFilter = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]]) / 273
    return cv2.filter2D(image, -1, gFilter)

def entropy_2d(image):
    entropy = 0
    kernel = np.ones((11,11), dtype = np.float32) / 121
    prob = np.zeros((256,256), dtype= np.float32)
    avg_gray = cv2.filter2D(image, -1, kernel)
    for i in range(256):
        for j in range(256):
            prob[i,j] = np.sum((image == i) & (avg_gray == j))
    prob = prob / (image.shape[0] * image.shape[1])
    for i in range(256):
        for j in range(256):
            entropy += prob[i,j] * np.log(prob[i,j] + 1e-10)
    return -entropy

def getResoluLv(g):
    if (g >= 192): return g/64 - 1
    elif (g >= 64): return g/128 + 0.5
    elif (g >= 32): return 3 - g/32
    else: return 6 - g/8

def gradFiltering(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (getResoluLv(image[i, j]) > 2):
                image[i, j] = 0
    return image
    

def compute_gradient_orientation(grad_x,grad_y):
    orientation = np.zeros(grad_x.shape)
    orientation[(np.abs(grad_x) > np.abs(TAN67_5*grad_y)) & (grad_x>0)] = DIRECTION.RIGHT
    orientation[(np.abs(grad_x) > np.abs(TAN67_5*grad_y)) & (grad_x<=0)] = DIRECTION.LEFT
    orientation[(np.abs(grad_x) < np.abs(TAN22_5*grad_y)) & (grad_y>0)] = DIRECTION.DOWN
    orientation[(np.abs(grad_x) < np.abs(TAN22_5*grad_y)) & (grad_y<=0)] = DIRECTION.UP
    diagonal = np.logical_or(np.abs(grad_x) <= np.abs(TAN67_5*grad_y), np.abs(grad_x) >= np.abs(TAN22_5*grad_y))
    orientation[diagonal & (grad_x>0) & (grad_y>0)] = DIRECTION.DOWN_RIGHT
    orientation[diagonal & (grad_x>0) & (grad_y<=0)] = DIRECTION.UP_RIGHT
    orientation[diagonal & (grad_x<=0) & (grad_y>0)] = DIRECTION.DOWN_LEFT
    orientation[diagonal & (grad_x<=0) & (grad_y<=0)] = DIRECTION.UP_LEFT
    
    return orientation

def find_anchor(grad, orientation):
    '''
    Return a list of anchor points coordinates (i,j), sorted by gradient value from large to small
    '''
    anchors = {}
    for i,j in np.ndindex(grad.shape):
        if i >= 1 and i < grad.shape[0]-1 and j >= 1 and j < grad.shape[1]-1:
            if orientation[i,j] == DIRECTION.RIGHT or orientation[i,j] == DIRECTION.LEFT:
                if grad[i,j] > grad[i,j-1] and grad[i,j] > grad[i,j+1]:
                    anchors[(i,j)] = grad[i,j]
            if orientation[i,j] == DIRECTION.DOWN or orientation[i,j] == DIRECTION.UP:
                if grad[i,j] > grad[i-1,j] and grad[i,j] > grad[i+1,j]:
                    anchors[(i,j)] = grad[i,j]
            if orientation[i,j] == DIRECTION.UP_LEFT or orientation[i,j] == DIRECTION.DOWN_RIGHT:
                if grad[i,j] > grad[i-1,j-1] and grad[i,j] > grad[i+1,j+1]:
                    anchors[(i,j)] = grad[i,j]
            if orientation[i,j] == DIRECTION.UP_RIGHT or orientation[i,j] == DIRECTION.DOWN_LEFT:
                if grad[i,j] > grad[i-1,j+1] and grad[i,j] > grad[i+1,j-1]:
                    anchors[(i,j)] = grad[i,j]
    # sort the anchors by gradient value from large to small
    anchors = dict(sorted(anchors.items(), key=lambda x: x[1], reverse=True))
    return list(anchors.keys())       

def curvature_prediction(grad, starting_point, starting_direction, all_anchors):
    '''
    Return a segement list of edge points
    '''
    edge_points = []
    segment_map = {}
    cur_direction = starting_direction
    i,j = starting_point
    
    while i >= 1 and i < grad.shape[0]-1 and j >= 1 and j < grad.shape[1]-1:
        # Loop detection
        if ((i,j) not in all_anchors) :
            break
        if (i,j) not in segment_map:
            segment_map[(i,j)] = [cur_direction]
            edge_points.append((i,j))
        else:
            if cur_direction in segment_map[(i,j)]:
                break
            else:
                segment_map[(i,j)].append(cur_direction)
        
        if cur_direction == DIRECTION.RIGHT:
            next = np.argmax(np.array([grad[i-1,j+1],grad[i,j+1],grad[i+1,j+1]]))
            if next == 0:
                cur_direction = DIRECTION.UP_RIGHT
                i -= 1
                j += 1
            elif next == 1:
                cur_direction = DIRECTION.RIGHT
                j += 1
            else:
                cur_direction = DIRECTION.DOWN_RIGHT
                i += 1
                j += 1
        elif cur_direction == DIRECTION.LEFT:
            next = np.argmax(np.array([grad[i-1,j-1],grad[i,j-1],grad[i+1,j-1]]))
            if next == 0:
                cur_direction = DIRECTION.UP_LEFT
                i -= 1
                j -= 1
            elif next == 1:
                cur_direction = DIRECTION.LEFT
                j -= 1
            else:
                cur_direction = DIRECTION.DOWN_LEFT
                i += 1
                j -= 1
        elif cur_direction == DIRECTION.UP:
            next = np.argmax(np.array([grad[i-1,j-1],grad[i-1,j],grad[i-1,j+1]]))
            if next == 0:
                cur_direction = DIRECTION.UP_LEFT
                i -= 1
                j -= 1
            elif next == 1:
                cur_direction = DIRECTION.UP
                i -= 1
            else:
                cur_direction = DIRECTION.UP_RIGHT
                i -= 1
                j += 1
        elif cur_direction == DIRECTION.DOWN:
            next = np.argmax(np.array([grad[i+1,j-1],grad[i+1,j],grad[i+1,j+1]]))
            if next == 0:
                cur_direction = DIRECTION.DOWN_LEFT
                i += 1
                j -= 1
            elif next == 1:
                cur_direction = DIRECTION.DOWN
                i += 1
            else:
                cur_direction = DIRECTION.DOWN_RIGHT
                i += 1
                j += 1
        elif cur_direction == DIRECTION.UP_RIGHT:
            next = np.argmax(np.array([grad[i-1,j],grad[i-1,j+1],grad[i,j+1]]))
            if next == 0:
                cur_direction = DIRECTION.UP
                i -= 1
            elif next == 1:
                cur_direction = DIRECTION.UP_RIGHT
                i -= 1
                j += 1
            else:
                cur_direction = DIRECTION.RIGHT
                j += 1
        elif cur_direction == DIRECTION.DOWN_RIGHT:
            next = np.argmax(np.array([grad[i,j+1],grad[i+1,j+1],grad[i+1,j]]))
            if next == 0:
                cur_direction = DIRECTION.RIGHT
                j += 1
            elif next == 1:
                cur_direction = DIRECTION.DOWN_RIGHT
                i += 1
                j += 1
            else:
                cur_direction = DIRECTION.DOWN
                i += 1
        elif cur_direction == DIRECTION.UP_LEFT:
            next = np.argmax(np.array([grad[i-1,j],grad[i-1,j-1],grad[i,j-1]]))
            if next == 0:
                cur_direction = DIRECTION.UP
                i -= 1
            elif next == 1:
                cur_direction = DIRECTION.UP_LEFT
                i -= 1
                j -= 1
            else:
                cur_direction = DIRECTION.LEFT
                j -= 1
        elif cur_direction == DIRECTION.DOWN_LEFT:
            next = np.argmax(np.array([grad[i,j-1],grad[i+1,j-1],grad[i+1,j]]))
            if next == 0:
                cur_direction = DIRECTION.LEFT
                j -= 1
            elif next == 1:
                cur_direction = DIRECTION.DOWN_LEFT
                i += 1
                j -= 1
            else:
                cur_direction = DIRECTION.DOWN
                i += 1
    if ((i,j) in all_anchors) : edge_points.append((i,j)) # Add the boundary point
    
    return edge_points

def edge_linking(grad, orientation, anchors, threshold_low, threshold_high, save_partial, save_path):
    edge_map = np.zeros(grad.shape, dtype = np.uint8)
    edge_count = 0
    for anchor in anchors:
        i,j = anchor
        if(edge_map[i,j] == 255):
            continue
        s1_dir, s2_dir = DIRECTION.get_perpendicular_direction(orientation[i,j])
        s1 = curvature_prediction(grad, (i,j), s1_dir, anchors)
        s2 = curvature_prediction(grad, (i,j), s2_dir, anchors)
        if len(s1) + len(s2) < threshold_low:
            continue
        s = s1 + s2
        for p in s:
            edge_map[p] = 255
            edge_count += 1
        if edge_count > threshold_high:
            break
        if save_partial:
            cv2.imwrite(os.path.join(save_path, "edge_map.png"), edge_map)
    return edge_map
        

def main():
    # args = parse_args()
    
    # if(args.input_path == None):
    #     print("Please provide input path")
    #     return
    
    # os.makedirs(args.output_path, exist_ok=True)
    np.random.seed(0)
    
    sample = GausBlur(cv2.imread("./SampleImages/lena.png", cv2.IMREAD_GRAYSCALE))
    
    # compute entropy and thresholding
    entropy = entropy_2d(sample)
    ## TODO: implement threshold
    threshold_high = 0.043 * sample.shape[0] * sample.shape[1]
    if (entropy > 12.7) : threshold_high = 0.087 * sample.shape[0] * sample.shape[1]
    elif (entropy >= 10.8) : threshold_high = 0.065 * sample.shape[0] * sample.shape[1]
    # compute gradient and orientation, then find anchor points
    grad_x = cv2.Sobel(sample, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(sample, cv2.CV_64F, 0, 1, ksize=3)
    cv2.normalize(grad_x, grad_x, -255, 255, cv2.NORM_MINMAX)
    cv2.normalize(grad_y, grad_y, -255, 255, cv2.NORM_MINMAX)
    grad = gradFiltering((np.abs(grad_x) + np.abs(grad_y))/2)
    # cv2.imwrite(os.path.join("./OutputImages", "grad.png"), grad)
    orientation = compute_gradient_orientation(grad_x, grad_y)
    anchors = find_anchor(grad, orientation)
    # blank = np.zeros(grad.shape, dtype = np.uint8)
    # for ac in anchors:
    #     blank[ac[0], ac[1]] = 255
    # cv2.imwrite(os.path.join("./OutputImages", "anchor.png"), blank)
    # find edge points
    edge_map = edge_linking(grad, orientation, anchors, 20, threshold_high, save_partial=True, save_path="./OutputImages")
    cv2.imwrite(os.path.join("./OutputImages", "edge_map.png"), edge_map)
    

if __name__ == "__main__":
    main()