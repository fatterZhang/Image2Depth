import time
from options.test_options import TestOptions
from dataloader.data_loader import dataloader
from util.visualizer import Visualizer
from util import html
from model.model import *

opt = TestOptions().parse()
opt.batchSize = 1
opt.nThreads = 1
opt.serial_batches = True
opt.no_flip = True

dataset = dataloader(opt)
dataset_size = len(dataset)
print ('testing images = %d ' % dataset_size)

model = Image2Depth()
model.initialize(opt)
visualizer = Visualizer(opt)

web_dir = os.path.join(opt.results_dir,opt.name, '%s_%s' %(opt.phase, opt.which_epoch))
web_page = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# testing
for i,data in enumerate(dataset):
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('process image...... %s' % img_path)
    visualizer.save_images(web_page, visuals, img_path)
