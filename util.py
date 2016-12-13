import skimage.io
import numpy

import sys
import os.path


def get_data_dir():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.abspath(script_dir+"/data")
    return data_dir

def get_image_dir():
    return get_data_dir()+"/images"

def debug(s):
    print(s,file=sys.stderr)

def read_grey_image(image_file):
    image = skimage.io.imread(image_file,as_grey=True)
    #Convert to float format
    if numpy.dtype('uint8') == image.dtype:
        image = image/255
    #Convert to 2 bit image
    f = numpy.vectorize(lambda x: 1 if x > 0.99 else 0)
    image = f(image)
    return image
