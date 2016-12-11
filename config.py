character_number = 26
network_name="wakaba_neural"
#random seed for neural network training
seed = 34
#fixed width and heigth of character samples
sample_h = 30
sample_w = 30
#dot size over 'i' 'j' characters
dot_size = 6
#regularization parameter to prevent overfitting
reg = 1
#range of values of weights in initial random initialization
w_range = 0.1
#method for scipy.optimize.minimize
method = "CG"

#Tweak this parameters for optimal results
#Also consider CAPTCHA_SCRIBBLE and CAPTCHA_ROTATION in gencaptcha.pl

#size of generated training set (by original perl script)
gen_train_size = 10000
#neural network's hidden layer size
hidden_layer = 100
#maximum number of learning iterations
maxiter = 300
