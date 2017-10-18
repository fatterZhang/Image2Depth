import os
import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from . import networks

class Image2Depth():
    def name(self):
        return 'Image2DepthModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_Image = self.Tensor(nb, opt.input_nc, size, size)
        self.input_depth = self.Tensor(nb, opt.output_nc, size, size)

        # define the networks
        self.netG_depth = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.Resblock,
                                            opt.norm, not opt.no_dropout, self.gpu_ids)
        self.netG_Image = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.Resblock,
                                            opt.norm, not opt.no_dropout, self.gpu_ids)

        if self.isTrain:
            self.netD_depth = networks.define_D(opt.output_nc, opt.ndf,opt.n_layers_D, opt.norm, self.gpu_ids)
            self.netD_Image = networks.define_D(opt.input_nc, opt.ndf, opt.n_layers_D, opt.norm, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_depth, 'G_depth', which_epoch)
            self.load_network(self.netG_Image, 'G_Image', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_depth, 'D_depth', which_epoch)
                self.load_network(self.netD_Image, 'D_Image', which_epoch)

        # define the loss function and optimizer
        if self.isTrain:
            self.old_G_depth_lr = opt.lr_G_depth
            self.old_G_Image_lr = opt.lr_G_Image
            self.old_D_depth_lr = opt.lr_D_depth
            self.old_D_Image_lr = opt.lr_D_Image

            self.fake_Image_pool = ImagePool(opt.pool_size)
            self.fake_depth_pool = ImagePool(opt.pool_size)

            # loss function
            self.criterionCycle = torch.nn.L1Loss()

            #initialize optimizers
            self.optimizer_G_depth = torch.optim.Adam(self.netG_depth.parameters(), lr=opt.lr_G_depth, betas=(opt.beta1, 0.999))
            self.optimizer_G_Image = torch.optim.Adam(self.netG_Image.parameters(), lr=opt.lr_G_Image, betas=(opt.beta1, 0.999))
            self.optimizer_D_depth = torch.optim.Adam(self.netD_depth.parameters(), lr=opt.lr_D_depth, betas=(opt.beta1, 0.999))
            self.optimizer_D_Image = torch.optim.Adam(self.netD_Image.parameters(), lr=opt.lr_D_Image, betas=(opt.beta1, 0.999))

        print('-------------------------Networks initialized---------------------------')
        networks.print_network(self.netG_depth)
        networks.print_network(self.netG_Image)
        if self.isTrain:
            networks.print_network(self.netD_depth)
            networks.print_network(self.netD_Image)
        print('-------------------------------------------------------------------------')

    def set_input(self, input):
        self.input = input
        Image2Depth = self.opt.which_direction == 'Image2Depth'
        input_Image = input['A' if Image2Depth else 'B']
        input_depth = input['B' if Image2Depth else 'A']
        self.input_Image.resize_(input_Image.size()).copy_(input_Image)
        self.input_depth.resize_(input_depth.size()).copy_(input_depth)
        self.image_paths = input['A_paths' if Image2Depth else 'B_paths']

    def forward(self):
        self.real_Image = Variable(self.input_Image)
        self.real_depth = Variable(self.input_depth)

    def test(self):
        self.real_Image = Variable(self.input_Image, volatile=True)
        self.fake_depth = self.netG_depth.forward(self.real_Image)
        self.rec_Image = self.netG_Image.forward(self.fake_depth)

        self.real_depth = Variable(self.input_depth, volatile=True)
        self.fake_Image = self.netG_Image.forward(self.real_depth)
        self.rec_depth = self.netG_depth.forward(self.fake_Image)

    def backward_D_basic(self, netD, real, fake):
        # real
        D_real = netD.forward(real)
        D_real_loss = torch.mean((D_real-1)**2)
        #fake
        D_fake = netD.forward(fake)
        D_fake_loss = torch.mean(D_fake**2)

        #lsGAN loss
        D_loss = 0.5 * (D_real_loss + D_fake_loss)

        D_loss.backward()

        return D_loss

    def backward_D_depth(self):
        fake_depth = self.fake_depth_pool.query(self.fake_depth)
        self.D_loss_depth = self.backward_D_basic(self.netD_depth, self.real_depth, fake_depth)

    def backward_D_Image(self):
        fake_Image = self.fake_Image_pool.query(self.fake_Image)
        self.D_loss_Image = self.backward_D_basic(self.netD_Image, self.real_Image, fake_Image)

    def backward_G_depth(self):
        lambda_Image = self.opt.lambda_Image
        lambda_smooth = self.opt.lambda_smooth
        # GAN loss
        self.fake_depth = self.netG_depth.forward(self.real_Image)
        D_fake = self.netD_depth.forward(self.fake_depth)
        self.G_loss_depth = 0.5 * torch.mean((D_fake-1)**2)
        # depth continue loss


        # forward cycle loss
        self.rec_Image = self.netG_Image.forward(self.fake_depth)
        self.cycle_loss_Image = self.criterionCycle(self.rec_Image, self.real_Image) * lambda_Image

        self.image2depth_loss = self.G_loss_depth + self.cycle_loss_Image

        self.image2depth_loss.backward()

    def backward_G_Image(self):
        lambda_depth = self.opt.lambda_depth
        #GAN loss
        self.fake_Image = self.netG_Image.forward(self.real_depth)
        D_fake = self.netD_depth.forward(self.fake_Image)
        self.G_loss_Image = 0.5 * torch.mean((D_fake-1)**2)

        #forward cycle loss
        self.rec_depth = self.netG_depth.forward(self.fake_Image)
        self.cycle_loss_depth = self.criterionCycle(self.rec_depth, self.real_depth) * lambda_depth

        self.depth2image_loss = self.G_loss_Image + self.cycle_loss_depth

        #self.depth2image_loss.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_depth
        self.optimizer_G_depth.zero_grad()
        self.backward_G_depth()
        self.optimizer_G_depth.step()
        # G_Image
        self.optimizer_G_Image.zero_grad()
        self.backward_G_Image()
        self.optimizer_G_Image.step()
        # D_depth
        self.optimizer_D_depth.zero_grad()
        self.backward_D_depth()
        self.optimizer_D_depth.step()
        # D_Image
        self.optimizer_D_Image.zero_grad()
        self.backward_D_Image()
        self.optimizer_D_Image.step()


    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device_id=gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))