#!/usr/bin/python3
import glob
import os.path
import re
import sys

import wkcaptcha
from util import debug
import util


def recognize_and_rename(dir):
    '''Recognizing all files file.gif in directory `dir` and renaming to <captcha>.gif.'''
    if not os.path.exists(dir):
        os.mkdir(dir)
    images = glob.glob(dir + "/*.gif")
    for image_file in images:
        captcha_p = wkcaptcha.predict_file(image_file)
        os.rename(image_file,dir +"/"+captcha_p+".gif")

def check_labeled_dir(NN,dir,limit=None,shift=0):
    '''Checking accuracy on <captcha>.gif files.'''
    total = 0
    recognized = 0
    if not os.path.exists(dir):
        os.mkdir(dir)
    images = glob.glob(dir + "/*.gif")
    for image_file in images[shift:]:
        total += 1
        captcha_p = wkcaptcha.predict_file(NN,image_file)
        captcha = re.match("(.*)\.gif",os.path.basename(image_file)).group(1)
        if(captcha == captcha_p):
            recognized += 1
        if(limit and total >= limit):
            break
    return recognized/total

if __name__ == '__main__':
    if(len(sys.argv) > 1):
        debug("Accuracy on captcha in {} directory: {}".format(sys.argv[1], check_labeled_dir(wkcaptcha.get_saved_classifier(),sys.argv[1])))
    else:
        debug("Accuracy on generated set: {}".format(check_labeled_dir(wkcaptcha.get_saved_classifier(),util.get_image_dir(),limit=100)))
