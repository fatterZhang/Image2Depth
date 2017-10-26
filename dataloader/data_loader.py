import os.path
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from  dataloader.image_folder import make_dataset
import random

def dataloader(opt):
    datasets = CreateDataset(opt);
    dataset = data.DataLoader(datasets,batch_size=opt.batchSize,
                              shuffle=opt.serial_batches,num_workers=int(opt.nThreads))
    return dataset


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'unaligned':
        dataset = UnalignedData()
    else:
        raise ValueError('Dataset [%s] not recognized.' % opt.dataset_mode)

    print ('dataset [%s] was created' % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class UnalignedData(data.Dataset):
    def initialize(self,opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot,opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot,opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transform_Image = get_transform(opt,True)
        self.transform_depth = get_transform(opt,False)

    def __getitem__(self, item):
        A_path = self.A_paths[item % self.A_size]
        index_B = random.randint(0,self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_image = Image.open(A_path)
        B_image = Image.open(B_path)

        A_image = self.transform_Image(A_image)
        B_image = self.transform_depth(B_image)

        return {'A':A_image,'B':B_image,'A_paths':A_path,'B_paths':B_path}

    def __len__(self):
        return max(self.A_size,self.B_size)

    def name(self):
        return 'UnalignedDataset'


def get_transform(opt,augment=False):
    transform_list = []
    if opt.resize_or_crop == "resize_and_crop":
        osize = [opt.loadSize,opt.loadSize]
        transform_list.append(transforms.Resize(osize,Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(lambda img:__scale_width(img,opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(lambda img:__scale_width(img,opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    if opt.isTrain and augment:
        brightness = random.uniform(0.8,1.2)
        contrast = random.uniform(0.5,2.0)
        saturation = random.uniform(0.8,1.2)
        transform_list.append(transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=0.5))

    transform_list += [transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]

    return transforms.Compose(transform_list)

def __scale_width(img,target_width):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width *oh/ow)
    return img.resize((w,h),Image.BICUBIC)
