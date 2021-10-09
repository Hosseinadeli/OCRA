# --------------------
# Data
# --------------------

import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import torch

def fetch_dataloader(args, batch_size, train=True, train_val_split='none', download=True):
    """
    load dataset depending on the task
    currently implemented tasks:
        -svhn
        -cifar10
        -mnist
        -multimnist, multimnist_cluttered 
    args
        -args
        -batch size
        -train: if True, load train dataset, else test dataset
        -train_val_split: 
            'none', load entire train dataset
            'train', load first 90% as train dataset
            'val', load last 10% as val dataset
            'train-val', load 90% train, 10% val dataset
    """
    kwargs = {'num_workers': 0, 'pin_memory': False} if args.device.type == 'cuda' else {}
    
    if args.task == 'svhn': 
        data_root = args.data_dir + '/svhn-data'
        #kwargs.pop('input_size', None)

        if args.num_targets == 1:
            if train:
                train_loader = torch.utils.data.DataLoader(datasets.SVHN(
                        root=data_root, split='train', download=download,transform=T.Compose([T.ToTensor(),
                            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])),batch_size=batch_size, shuffle=True, **kwargs)
                return train_loader

            else:
                test_loader = torch.utils.data.DataLoader(datasets.SVHN(
                        root=data_root, split='test', download=download,transform=T.Compose([T.ToTensor(),
                            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])), batch_size=batch_size, shuffle=False, **kwargs)
                return test_loader

                
    elif args.task == 'cifar10': 

        data_root  = args.data_dir + '/cifar10-data'    
        #kwargs.pop('input_size', None)
        transform = T.Compose([T.ToTensor()])  # transforms.RandomCrop(size=32, padding=shift_pixels),
        
        if train: 
            trainset = datasets.CIFAR10(root=data_root, train=True,download=download, transform=transform)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, **kwargs)
            return train_loader
        
        else: 
            testset = datasets.CIFAR10(root=data_root, train=False, download=download, transform=transform)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, **kwargs)
            return test_loader

    elif (args.task == 'mnist_ctrv') or (args.task == 'mnist'): 
        transforms = T.Compose([T.ToTensor()])
        dataset = datasets.MNIST(root=args.data_dir, train=train, download=download, transform=transforms)

        if train: 
            train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
            train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=train, drop_last=True, **kwargs)
            val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
            return train_dataloader, val_dataloader
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    
    else:
        
        # 2 mnist digits overlapping 
        if args.task == 'multimnist':
            data_root = args.data_dir + 'multimnist/'
#             train_datafile = 'mnist_overlap4pix_nodup_20fold_36_train.pt'
#             test_datafile = 'mnist_overlap4pix_nodup_20fold_36_test.pt'
            train_datafile = 'mnist_overlap4pix_nodup_50fold_36_train.npz'
            test_datafile = 'mnist_overlap4pix_nodup_50fold_36_test.npz'
        
        # 2 mnist digits on a 100*100 canvas with 6 pieces of clutter 
        elif args.task == 'multimnist_cluttered': 
            data_root = args.data_dir + 'multimnist/'
            train_datafile = 'mnist_cluttered2o6c_3fold_100_trainval.pt'
            test_datafile = 'mnist_cluttered2o6c_3fold_100_test.pt'
            
        # 3 mnist digits without category duplicate on a 100*100 canvas
        elif args.task == 'multimnist-3o':
            data_root = args.data_dir + 'multimnist/'
            train_datafile = 'mnist_100x100_multi3_nodup_train.pt'
            test_datafile = 'mnist_100x100_multi3_nodup_test.pt'
                    
        # 1 mnist digit on a 100*100 canvas with 8 pieces of clutter      
        elif args.task == 'mnist_cluttered': 
            data_root = args.data_dir + 'multimnist/'
            train_datafile = 'mnist_60x60_cluttered1o4c_train.pt'
            test_datafile = 'mnist_60x60_cluttered1o4c_test.pt'            
            
        elif args.task == 'multisvhn': 
            data_root = args.data_dir + '/svhn-data/'
            train_datafile = 'multisvhn_train_14classes.pt'
            test_datafile = 'multisvhn_test_14classes.pt'
            
        if train: # load train dataset
            # if the datafile is in pytorch format, just load it, if numpy then load and convert to tensors
            if train_datafile[-2:] == 'pt':
                tensor_trainval_ims, tensor_trainval_ys = torch.load(data_root+train_datafile)
            else:
                trainval_ims_ys = np.load(data_root+train_datafile)
                tensor_trainval_ims = torch.Tensor(trainval_ims_ys['tensor_train_ims'])
                tensor_trainval_ys = torch.Tensor(trainval_ims_ys['tensor_train_ys'])
            
            if train_val_split == 'none':  
                #return the entire training set for training 
                train_dataset = TensorDataset(tensor_trainval_ims,tensor_trainval_ys) # create your datset
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, **kwargs) # create your dataloader
                return train_dataloader
            
            else: 
                #split 90/10 training validation 
                tensor_train_ims = tensor_trainval_ims[0:int(.9*len(tensor_trainval_ims))]
                tensor_train_ys = tensor_trainval_ys[0:int(.9*len(tensor_trainval_ys))]
                tensor_val_ims = tensor_trainval_ims[int(.9*len(tensor_trainval_ims)):len(tensor_trainval_ims)]
                tensor_val_ys = tensor_trainval_ys[int(.9*len(tensor_trainval_ys)):len(tensor_trainval_ys)]
                
                if train_val_split == 'train':
                    
                    train_dataset = TensorDataset(tensor_train_ims,tensor_train_ys) # create your datset
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, **kwargs) # create your dataloader
                    return train_dataloader

                elif train_val_split == 'val':

                    val_dataset = TensorDataset(tensor_val_ims,tensor_val_ys) # create your datset
                    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, **kwargs) # create your dataloader
                    return val_loader
                
                elif train_val_split == 'train-val':                
                    
                    train_dataset = TensorDataset(tensor_train_ims,tensor_train_ys) # create your datset
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, **kwargs) # create your dataloader
                    val_dataset = TensorDataset(tensor_val_ims,tensor_val_ys) # create your datset
                    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, **kwargs) # create your dataloader
                    return train_dataloader, val_dataloader
                
  
        elif not train: # load test dataset
            # if the datafile is in pytorch format, just load it, if numpy then load and convert to tensors
            if test_datafile[-2:] == 'pt':
                tensor_test_ims, tensor_test_ys = torch.load(data_root+test_datafile)
            else:
                test_ims_ys = np.load(data_root+test_datafile)
                tensor_test_ims = torch.Tensor(test_ims_ys['tensor_test_ims'])
                tensor_test_ys = torch.Tensor(test_ims_ys['tensor_test_ys'])
            
            test_dataset = TensorDataset(tensor_test_ims,tensor_test_ys) # create your datset
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, **kwargs) # create your dataloader
        
            return test_dataloader