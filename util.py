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