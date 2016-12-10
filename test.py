#!/usr/bin/python3
import glob
import os.path
import re

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

def check_labeled_dir(dir):
	'''Checking accuracy on <captcha>.gif files.'''
	total = 0
	recognized = 0
	if not os.path.exists(dir):
		os.mkdir(dir)
	images = glob.glob(dir + "/*.gif")
	for image_file in images:
		total += 1
		captcha_p = wkcaptcha.predict_file(image_file)
		captcha = re.match("(.*)\.gif",os.path.basename(image_file)).group(1)
		if(captcha == captcha_p):
			recognized += 1
	return recognized/total

def selfcheck():
	return check_labeled_dir(util.get_image_dir())

if __name__ == '__main__':
	debug("Checking whole full captcha files.")
	debug("Accuracy on generated set: {}".format(selfcheck()))