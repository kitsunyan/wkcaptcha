import numpy

import os.path
import glob
import re

import config
import util
import split

def components(arr):
    '''Finds connected components of 2d array. Returns (mask,component_number) where:
    mask: 2d array of the same size, where non-empty cell is marked by corresponding component number
    component_number: number of connected components'''
    def empty(k):
        return k == 1
    def dfs(x,y):
        #check if (x,y) is inside array
        if(not (0 <= x and x < arr.shape[0] and 0 <= y and y < arr.shape[1])):
            return
        #check if (x,y) should be visited
        if(empty(arr[x][y])):
            return
        #check if (x,y) was not already visited
        if(mask[x][y] != 0):
            return
        #visit (x,y)
        mask[x][y] = component_number
        dfs(x-1,y)
        dfs(x+1,y)
        dfs(x,y+1)
        dfs(x,y-1)
        #dfs(x-1,y+1)
        #dfs(x+1,y+1)
        #dfs(x-1,y-1)
        #dfs(x+1,y-1)

    mask = numpy.zeros(shape=arr.shape,dtype=numpy.uint8)
    component_number = 0
    for y in range(arr.shape[1]):
        for x in range(arr.shape[0]):        
            if( (not empty(arr[x][y])) and (mask[x][y] == 0)):
                component_number += 1
                dfs(x,y)                
    return (mask,component_number)

def box_coordinates(mask,num_of_components):
    '''Finds rectangulars in which cells located in one connected component can be placed.
     Returns list of 4-tuples in (x1,x2,y1,y2) format'''
    def x1(m):
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if(mask[x][y] == m):
                    return x

    def x2(m):
        for x in reversed(range(mask.shape[0])):
            for y in range(0,mask.shape[1]):
                if(mask[x][y] == m):
                    return x

    def y1(m):
        for y in range(mask.shape[1]):
            for x in range(mask.shape[0]):
                if(mask[x][y] == m):
                    return y

    def y2(m):
        for y in reversed(range(mask.shape[1])):
            for x in range(mask.shape[0]):
                if(mask[x][y] == m):
                    return y
                    
    boxes = []
    for m in range(1,num_of_components+1):
        boxes.append((x1(m),x2(m),y1(m),y2(m)))
    return boxes

def cut_boxes(mask,boxes):
    '''Cuts rectangualar from 2d array containing elements from corresponding connected component only'''
    cutted = []
    for c in range(len(boxes)):
        x1,x2,y1,y2 = boxes[c]
        box = numpy.zeros(shape=(x2-x1+1,y2-y1+1),dtype=numpy.uint8)
        for x in range(x1,x2+1):
            for y in range(y1,y2+1):
                #c-th box contains (c+1)th component
                if(mask[x][y] == c+1):
                    box[x-x1][y-y1] = 1
        cutted.append(box)
    return cutted

def filter_dots(shape,box_coord):
    '''filtering out dots over "i" and "j" letters by replacing with empyt box'''
    def is_dot(z):
        return (z[1]-z[0])+1 <= config.dot_size and z[1] < shape[0]*0.5
    return list(map(lambda z: (0,0,0,0) if is_dot(z) else z,box_coord))

def filter_nulls(segments):
    return list(filter(lambda x: x.shape[0] > 1,segments))

def segment_image(image):
    '''Segments image into rectangular parts, each containing connected component'''
    mask,num_of_components = components(image)
    box_coord = box_coordinates(mask,num_of_components)
    box_coord = filter_dots(image.shape,box_coord)
    segments = cut_boxes(mask,box_coord)
    segments = filter_nulls(segments)
    return segments

def var_to_fixed(v_sgm):
    '''Converts variable size 2d segment of characters into centered fixed size 1d segment'''
    h_v = v_sgm.shape[0]
    w_v = v_sgm.shape[1]
    assert(config.sample_h > h_v)
    assert(config.sample_w > w_v)
    shift_h = int((config.sample_h - h_v)/2)
    shift_w = int((config.sample_w - w_v)/2)
    f_sgm = numpy.zeros(shape=(config.sample_h,config.sample_w),dtype=numpy.uint8)
    for i in range(h_v):
        for j in range(w_v):
            f_sgm[i+shift_h][j+shift_w]=v_sgm[i][j]
    return f_sgm.flatten()

def image_to_features(image,captcha):
    '''Extracts feature and label matrix from image.'''
    image_segments = segment_image(image)   
    if(len(captcha) != len(image_segments)):
        return (numpy.zeros((0,config.sample_h*config.sample_w),dtype=numpy.uint8),numpy.array([],dtype=numpy.uint8))
    X=numpy.array(list(map(var_to_fixed,image_segments)))   
    y = numpy.array(list(map(lambda c:ord(c)-ord(config.first_character),captcha)))
    return (X,y)    

def extract_features():
    '''Extract features from all labeled images.'''
    image_dir = util.get_image_dir()
    images = glob.glob(image_dir+"/*.gif")
    characters = []
    var_segments = []

    def extract_single(image_file):
        captcha = re.match("(.*)\.gif",os.path.basename(image_file)).group(1)
        image = util.read_grey_image(image_file)
        return image_to_features(image,captcha)

    X,y = list(zip(*list(map(extract_single,images))))
    #return X,y
    X = numpy.concatenate(X,axis=0)
    y = numpy.concatenate(y,axis=0)
    return (X,y)

def split_segments(image_segments):
    '''Splitting of crossint characters.
    image_segments: list of character segments
    return value: list of splitted characters'''
    image_segments_2 = []
    for segment in image_segments:
            if(segment.shape[1] >= config.double_character_width):
                splitted = split.split_joint(segment, letters = 2)
                image_segments_2 += splitted
            else:
                image_segments_2.append(segment)
    return image_segments_2
