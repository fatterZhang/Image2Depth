import time
from options.train_options import TrainOptions
from dataloader.data_loader import dataloader
from util.visualizer import Visualizer

opt = TrainOptions().parse()
dataset = dataloader(opt)

for epoch in range(opt.epoch_count,opt.niter + opt.niter_decay+1):
    epoch_start_time = time.time()
    epoch_iter = 0

    # training
    for i,data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        print data