import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
import utils_fixation
import PIL


def preprocess(img, out_size=None, data='img'):
    transformations = []
    if data in ('img', 'sal'):
        transformations.append(transforms.Resize(
            out_size, interpolation=PIL.Image.LANCZOS))
    else:
        transformations.append(transforms.Resize(
            out_size, interpolation=PIL.Image.NEAREST))
    processing = transforms.Compose(transformations)
    tensor = processing(img)
    return tensor

class ImageData(data.Dataset):

    def __init__(self, img_root, label_root, transform, t_transform, f_transform, filename=None, mode='Train'):
        if filename is None:
            self.image_path = list(map(lambda x: os.path.join(img_root, x), os.listdir(img_root)))
            if mode == 'train':
                self.label_path = list(
                    map(lambda x: os.path.join(label_root, x.split('/')[-1][:-3] + 'png'), self.image_path))
            elif mode == 'test':
                self.label_path = []
                self.fix_path = []
            else:
                raise NotImplementedError
        else:
            lines = [line.rstrip('\n') for line in open(filename)]
            self.image_path = list(map(lambda x: os.path.join(img_root, x.split(' ')[0]), lines))
            if mode == 'train':
                self.label_path = list(map(lambda x: os.path.join(img_root, x.split(' ')[1]), lines))
                self.fix_path = list(map(lambda x: os.path.join(img_root, x.split(' ')[2]), lines))
            elif mode == 'test':
                self.label_path = []
                self.fix_path = []
            else:
                raise NotImplementedError
            # print(self.fix_path[0])

        self.transform = transform
        self.t_transform = t_transform
        self.f_transform = f_transform

    def __getitem__(self, item):
        image = Image.open(self.image_path[item]).convert('RGB')
        label = Image.open(self.label_path[item]).convert('L')

        if 'SALICON' in self.image_path[item] and self.fix_path is not None:
            fixation = utils_fixation.get_salicon_fixation_map(self.fix_path[item])
            fixation = Image.fromarray(fixation)

        elif 'CAT2000' in self.image_path[item] and self.fix_path is not None:
            fixation = utils_fixation.get_cat2000_fixation_map(self.fix_path[item])
            # fixation = Image.fromarray(fixation)

        elif 'MIT1003' or 'MIT300' in self.image_path[item] and self.fix_path is not None:
            fixation = Image.open(self.fix_path[item]).convert('L')
        
        elif 'DHF1K' in self.image_path[item] and self.fix_path is not None:
            fixation = Image.open(self.fix_path[item]).convert('L')

        elif 'UCFSPORTS' in self.image_path[item] and self.fix_path is not None:
            fixation = Image.open(self.fix_path[item]).convert('L')

        elif 'PseudoSaliency_avg' in self.image_path[item] and self.fix_path is not None:
            fixation = Image.open(self.fix_path[item]).convert('L')    
            
        elif 'DUT-OMRON' in self.image_path[item] and self.fix_path is not None:
            fixation = Image.open(self.fix_path[item]).convert('L') 
            
        elif 'PASCAL-S' in self.image_path[item] and self.fix_path is not None:
            fixation = Image.open(self.fix_path[item]).convert('L') 
            
        elif 'TORONTO' in self.image_path[item] and self.fix_path is not None:
            fixation = Image.open(self.fix_path[item]).convert('L')  

        else:
            fixation = None

        shape = image.size  # [w, h]
        if shape[0]/shape[1]>0.8:
            new_w = 384
            new_h = 224
        else:
            new_w = 224
            new_h = 384
        
        image=preprocess(image, (new_h, new_w), data='img')
        nonfixation=fixation

        if self.transform is not None:
            image = self.transform(image)
        if self.t_transform is not None:
            label = self.t_transform(label)
            fixation = self.f_transform(fixation)
            nonfixation =self.f_transform(nonfixation)

        return image, label, fixation, nonfixation

    def __len__(self):
        return len(self.image_path)


class ImageDataTest(data.Dataset):
    """ image dataset
    img_root:    image root (root which contain images)
    label_root:  label root (root which contains labels)
    transform:   pre-process for image
    t_transform: pre-process for label
    filename:    MSRA-B use xxx.txt to recognize train-val-test data (only for MSRA-B)
    """

    def __init__(self, img_root, label_root, transform, t_transform, filename=None, mode='Train'):
        if filename is None:
            self.image_path = list(map(lambda x: os.path.join(img_root, x), os.listdir(img_root)))
            self.label_path = list(
                map(lambda x: os.path.join(label_root, x.split('/')[-1][:-3] + 'png'), self.image_path))
        else:
            lines = [line.rstrip('\n') for line in open(filename)]
            self.image_path = list(map(lambda x: os.path.join(img_root, x.split(' ')[0]), lines))

        self.transform = transform
        self.t_transform = t_transform

    def __getitem__(self, item):
        image = Image.open(self.image_path[item]).convert('RGB')

        shape = image.size  # [w, h]
        if shape[0]/shape[1]>0.8:
            new_w = 384
            new_h = 224
        else:
            new_w = 224
            new_h = 384

        image = image.resize((new_w, new_h))

        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_path)


def get_loader_test(img_root, label_root, img_size, batch_size, filename=None, mode='test', num_thread=4, pin=True):
    t_transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Lambda(utils.normalize_tensor)
        # transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
    ])
    transform = transforms.Compose([
        # transforms.Resize((img_size, img_size)),
        # transforms.Resize((240, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageDataTest(img_root, label_root, None, None, filename=filename, mode='test')
    # data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_thread,
    #                               pin_memory=pin)
    return dataset

def select_nonsalient_points(label, fixation):
    label_numpy = np.asarray(label)  # [0, 255]
    fixation_numpy = np.asarray(fixation)  # [0, 255]
    mask_salient = ((label_numpy > np.mean(label_numpy)) + (fixation_numpy > 0))  # True or False
    mask_nonsalient = (mask_salient <= 0.0)  # True or False
    num_salpoint = np.sum((fixation_numpy > 0))
    index_nonsalient = np.where(mask_nonsalient)
    selected_index = np.random.randint(0, len(index_nonsalient[0]), num_salpoint)
    points_nonsal_x, points_nonsal_y = (index_nonsalient[0][selected_index], index_nonsalient[1][selected_index])  # (x, y)
    nonfixation_numpy=np.zeros_like(fixation_numpy)
    for cord in zip(points_nonsal_x, points_nonsal_y):
        nonfixation_numpy[cord] = 255
        # print(cord)
    show_image=False
    if show_image:
        plt.figure(0)
        plt.imshow(label)
        plt.figure(1)
        plt.imshow(fixation)
        plt.figure(2)
        plt.imshow(mask_salient)
        plt.figure(3)
        plt.imshow(mask_nonsalient)
        plt.figure(4)
        plt.imshow(nonfixation_numpy)
        plt.show()
    return nonfixation_numpy

# get the dataloader (Note: without data augmentation)
def get_loader(img_root, label_root, img_size, batch_size, filename=None, mode='train', num_thread=1, pin=True):
    if mode == 'train':
        transform = transforms.Compose([
            # transforms.Resize((img_size, img_size)),
            # transforms.Resize((240, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        t_transform = transforms.Compose([
            # transforms.Resize((img_size, img_size)),
            # transforms.Resize((240, 320)),
            transforms.ToTensor(),
            # transforms.Lambda(utils.normalize_tensor)
            # transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
        f_transform = transforms.Compose([
            # transforms.Resize((img_size, img_size)),
            # transforms.Resize((240, 320)),
            transforms.ToTensor(),
            transforms.Lambda(lambda fix: torch.gt(fix, 0.5))
        ])


        dataset = ImageData(img_root, label_root, transform, t_transform, f_transform, filename=filename, mode='train')
        data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_thread,
                                      pin_memory=pin)
        return data_loader
    else:
        t_transform = transforms.Compose([
            transforms.ToTensor()
            # transforms.Lambda(utils.normalize_tensor)
            # transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
        dataset = ImageData(img_root, label_root, None, t_transform, None, filename=filename, mode='test')
        # data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_thread,
        #                               pin_memory=pin)
        return dataset

