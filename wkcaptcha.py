#!/usr/bin/python3
import numpy

import sys
import os
import re
import glob
import subprocess

import config
import neural
import segment
import grammar
import util
from util import debug
import test


def generate_captcha(num):
    '''Generates labeled images using original captcha perl script from wakaba.'''
    current_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = util.get_data_dir()
    image_dir = data_dir + "/images"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    gen_script = os.path.abspath(script_dir+"/gencaptcha.pl")
    os.chdir(image_dir)
    for i in range(0,num):
        subprocess.call(["perl",gen_script,str(i)])
    os.chdir(current_dir)

def get_saved_classifier():
    '''Getting classifier from saved data if possible or training/generating new'''
    data_dir = util.get_data_dir()
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    #Trying to load saved neural network coefficients
    if os.path.exists(data_dir+"/"+config.network_name+".npy"):
        input_layer = config.sample_h * config.sample_w
        outer_layer = config.character_number
        neural_classifier = neural.NeuralClassfier(input_layer,config.hidden_layer,outer_layer,config.reg,random_seed=config.seed)
        neural_classifier.weights = numpy.load(data_dir+"/"+config.network_name+".npy")
        return neural_classifier

    #Trying to load saved training data
    if(os.path.exists(data_dir+"/X.npy") and os.path.exists(data_dir+"/y.npy")):
        X=numpy.load(data_dir+"/X.npy")
        y=numpy.load(data_dir+"/y.npy")
    else:
        image_dir = util.get_image_dir()
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
        if len(glob.glob(image_dir+"/*.gif")) < config.gen_train_size/2:
            debug("Generating recognized captcha images using perl script.")
            generate_captcha(config.gen_train_size)
        debug("Saved training data is not found. Generating new by segmentating images.")
        X,y = segment.extract_features()
        numpy.save(data_dir+"/X.npy",X)
        numpy.save(data_dir+"/y.npy",y)

    debug("Network coefficients are not found. Training new neural network.")
    neural_classifier = neural.train_network(X,y)
    numpy.save(data_dir+"/"+config.network_name+".npy",neural_classifier.weights)
    debug("Selfchecking full captcha files.")
    accuracy = test.check_labeled_dir(neural_classifier,util.get_image_dir(),limit=100)
    debug("Accuracy on generated set: {}".format(accuracy))
    return neural_classifier

def predict_image(NN,image):
    image_segments=segment.segment_image(image)
    image_segments=segment.split_segments(image_segments)
    captcha = ""
    for sgm in image_segments:
        if(sgm.shape[1] >= config.sample_w):
            captcha += "__"
        else:
            X=segment.var_to_fixed(sgm)
            captcha += (chr(ord(config.first_character)+neural.one_vs_all_to_class_number(NN.predict(X))))
    return captcha

def predict_image_many(NN,image):
    '''Give many uncertain variants with probabilities.
    Return format: list of (captcha,probability)'''
    def product(captchas,chars):
        #print(chars)
        if len(chars) == 1:
            for n in range(len(captchas)):
                captchas[n]=(captchas[n][0]+chars[0][0],captchas[n][1])
            return captchas
        new_captchas = []
        for captcha in captchas:
            for char in chars:
                new_captchas.append((captcha[0]+char[0],captcha[1]*char[1]))
        return new_captchas


    image_segments=segment.segment_image(image)
    image_segments=segment.split_segments(image_segments)
    captchas = [("",1)]
    for sgm in image_segments:
        if(sgm.shape[1] >= config.sample_w):
            next_chars = ["__"]
        else:
            X=segment.var_to_fixed(sgm)
            P = NN.predict(X)
            next_chars = [ (t[1],t[2]) for t in list(zip(P[0]>max(P[0])*config.uncertainty_coefficient,[chr(n+ord(config.first_character)) for n in range(26)],P[0])) if t[0] ]
        captchas = product(captchas,next_chars)

    return captchas


def predict_file(NN,image_file):
    image = util.read_grey_image(image_file)
    if config.use_grammar:
        captchas = predict_image_many(NN,image)
        if(len(captchas)>1):
            captchas = list(filter(lambda c:grammar.wakaba_accept(c[0]),captchas))
        if(len(captchas)>0):
            captcha = max(captchas,key=lambda x:x[1])[0]
        else:
            captcha = predict_image(NN,image)
    else:
        captcha = predict_image(NN,image)
    return captcha

if __name__ == '__main__':
    if(len(sys.argv) > 1):
        NN = get_saved_classifier()
        print(predict_file(NN,sys.argv[1]))
    else:
        print("Usage: wkcaptcha.py [FILE|URL]")
