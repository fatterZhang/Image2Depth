import time
from options.train_options import TrainOptions
from dataloader.data_loader import dataloader
from util.visualizer import Visualizer
from model.model import *

opt = TrainOptions().parse()
dataset = dataloader(opt)
dataset_size = len(dataset)
print ('training images = %d ' % dataset_size)

model = Image2Depth()
model.initialize(opt)
visualizer = Visualizer(opt)
total_steps = 0

for epoch in range(opt.epoch_count,opt.niter + opt.niter_decay+1):
    epoch_start_time = time.time()
    epoch_iter = 0

    # training
    for i,data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()