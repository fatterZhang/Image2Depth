import torch
import torch.nn as nn
import functools

# custom weights initialization called on generator and discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0,0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d,affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d,affine=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

# define the generator and initialize
def define_G(input_nc, output_nc, ngf, layers=[2,2,2,2], norm='batch', use_dropout=False,gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    netG = _netG_Unet(input_nc,output_nc,ngf=ngf,layers=layers,norm_layer=norm_layer,use_dropout=use_dropout,gpu_ids=gpu_ids)

    if use_gpu:
        assert (torch.cuda.is_available())
        netG.cuda()

    netG.apply(weights_init)

    return netG

# define the discriminator and initialize
def define_D(input_nc,ndf, which_model_netD, layers=[2,2,2,2],norm='batch',gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'Dcnet':
        netD = _netD(input_nc,ndf,layers[0],norm_layer,gpu_ids)
    elif which_model_netD =='Resnet':
        netD = _netD_Res(input_nc,ndf,layers,norm_layer,gpu_ids)
    else:
        raise NotImplementedError('No such discriminator [%s] is found' % which_model_netD)

    if use_gpu:
        assert (torch.cuda.is_available())
        netD.cuda()

    netD.apply(weights_init)

    return netD

# define print network
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    print net

    print ('Total number of parameters: %d' % num_params)

# Define the Unet
class _netG_Unet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, layers=[2,2,2,2], norm_layer=nn.BatchNorm2d, use_dropout=False,
                 gpu_ids=[], padding_type='reflect'):
        super(_netG_Unet, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        encoder = [nn.ReplicationPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, stride=2, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU()
                 ]

        unet_block = UnetBlock (ngf*8,ngf*8,blocks=layers[3],norm_layer=norm_layer,innermost=True)
        unet_block = UnetBlock (ngf*4,ngf*4,blocks=layers[2],submodule=unet_block,norm_layer=norm_layer)
        unet_block = UnetBlock (ngf*2,ngf*2,blocks=layers[1],submodule=unet_block,norm_layer=norm_layer)
        unet_block = UnetBlock (ngf,ngf,blocks=layers[0],submodule=unet_block,norm_layer=norm_layer,outermost=True)

        decoder = [nn.ReLU(),
                nn.ConvTranspose2d(ngf,ngf,kernel_size=4,stride=2,padding=1),
                norm_layer(ngf),
                nn.ReLU(),
                nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding= 1),
                nn.Tanh()
        ]

        model = encoder + [unet_block] + decoder

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and len(self.gpu_ids):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

# define the unet block
class UnetBlock(nn.Module):
    def __init__(self,input_nc,output_nc,blocks =2, submodule=None, innermost=False, outermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if outermost:
            self.input_nc = input_nc
            self.conv = nn.Conv2d(output_nc * 2, output_nc, kernel_size=3, stride=1, padding=1)
        else:
            self.input_nc = input_nc * 2
            self.conv = nn.Conv2d(output_nc * 3, output_nc, kernel_size=3, stride=1, padding=1)

        encoder = self._make_layer(Bottlenck,output_nc,blocks,norm_layer,use_bias,use_dropout,stride=2)

        if innermost:
            decoder = [
                nn.ConvTranspose2d(input_nc*4, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(output_nc),
            ]
            model = [encoder] + decoder
        else:
            decoder = [
                nn.ReLU(True),
                nn.ConvTranspose2d(input_nc*2, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(output_nc)
            ]
            if use_dropout:
                model = [encoder] + [submodule] + decoder + [nn.Dropout(0.5)]
            else:
                model = [encoder] + [submodule] + decoder

        self.model = nn.Sequential(*model)
        self.relu = nn.ReLU(True)

    def _make_layer(self,block,output_nc,blocks,norm_layer,use_bias,use_dropout,stride=1):
        downsample = None
        if stride != 1 or self.input_nc != output_nc*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.input_nc,output_nc*block.expansion,kernel_size=1,stride=stride,bias=False),
                norm_layer(output_nc*block.expansion)
            )

        layer = [block(self.input_nc,output_nc,norm_layer,use_bias,use_dropout,stride,downsample)]
        self.input_nc = output_nc*block.expansion

        for i in range(1,blocks):
            layer +=[block(self.input_nc,output_nc,norm_layer,use_bias,use_dropout)]

        return nn.Sequential(*layer)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            out = torch.cat([x, self.model(x)], 1)
            out = self.relu(out)
            out = self.conv(out)
            return out

# define a ResidualBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_nc, output_nc, norm_layer, use_bias, use_dropout=False, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        conv_block = [
                nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=stride, padding=1, bias=use_bias),
                norm_layer(output_nc),
                nn.ReLU(True)
        ]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [
                nn.Conv2d(output_nc, output_nc, kernel_size=3,padding=1, bias=use_bias),
                norm_layer(output_nc)
        ]

        self.conv_block = nn.Sequential(*conv_block)
        self.relu = nn.Sequential(nn.ReLU(True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv_block(x) + residual
        out = self.relu(out)

        return out

# define a bottleneck
class Bottlenck(nn.Module):
    expansion = 4

    def __init__(self, input_nc, output_nc, norm_layer, use_bias, use_dropout=False, stride=1, downsample=None):
        super(Bottlenck,self).__init__()
        conv_block = [
            nn.Conv2d(input_nc, output_nc, kernel_size=1, bias=use_bias),
            norm_layer(output_nc),
            nn.LeakyReLU(0.2, True)
        ]
        conv_block +=[
            nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=stride, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nn.LeakyReLU(0.2, True)
        ]
        conv_block +=[
            nn.Conv2d(output_nc, output_nc*4, kernel_size=1, bias=use_bias),
            norm_layer(output_nc*4)
        ]

        self.conv_block = nn.Sequential(*conv_block)
        self.relu = nn.Sequential(nn.LeakyReLU(0.2,True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv_block(x) + residual
        out = self.relu(out)

        return out

class _netD_Res(nn.Module):
    def __init__(self,input_nc,ndf=64,layers=[2,2,2,2],norm_layer = nn.BatchNorm2d,gpu_ids=[]):
        super(_netD_Res,self).__init__()
        self.gpu_ids = gpu_ids
        self.input_nc = ndf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence =[
            nn.Conv2d(input_nc,ndf,4,2,1,bias=False),
            nn.LeakyReLU(0.2,True)
        ]

        for n in range(len(layers)):
            sequence += self._make_layer(Bottlenck, ndf * (2**n), layers[n], norm_layer, use_bias, use_dropout=False, stride =2)

        sequence += [nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False)]

        self.model = nn.Sequential(*sequence)

    def _make_layer(self, block, output_nc, blocks, norm_layer, use_bias, use_dropout, stride=1):
        downsample = None
        if stride != 1 or self.input_nc != output_nc * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.input_nc, output_nc * block.expansion, kernel_size=1, stride=stride, bias=False),
                    norm_layer(output_nc * block.expansion)
            )

        layer = [block(self.input_nc, output_nc, norm_layer, use_bias, use_dropout, stride, downsample)]
        self.input_nc = output_nc * block.expansion

        for i in range(1, blocks):
            layer += [block(self.input_nc, output_nc, norm_layer, use_bias, use_dropout)]

        return layer

    def forward(self,input):
        if isinstance(input.data,torch.cuda.FloatTensor) and len(self.gpu_ids):
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)

        return output.view(-1,1).squeeze(1)



# Define the Discriminator layer
class _netD(nn.Module):
    def __init__(self,input_nc,ndf=64,n_layer=3,norm_layer = nn.BatchNorm2d,gpu_ids =[]):
        super(_netD,self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [
            nn.Conv2d(input_nc,ndf,4,2,1,bias=False),
            nn.LeakyReLU(0.2,True)
        ]

        nf_mult = 1
        for n in range(1,n_layer):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n,8)
            sequence +=[
                nn.Conv2d(ndf*nf_mult_prev,ndf*nf_mult,4,2,1,bias=use_bias),
                norm_layer(ndf*nf_mult),
                nn.LeakyReLU(0.2,True)
            ]

        sequence +=[nn.Conv2d(ndf*nf_mult, 1, 4, 1, 0, bias=False)]

        self.model = nn.Sequential(*sequence)

    def forward(self,input):
        if isinstance(input.data,torch.cuda.FloatTensor) and len(self.gpu_ids):
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)

        return output.view(-1,1).squeeze(1)