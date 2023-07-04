import os
import torch

from lib.utils import load_class_names
from datasets.custom_dataset import CustomDataset
from datasets.UCASAOD_dataset import UCASAODDataset
from datasets.DOTA_dataset import DOTADataset


def load_data(data_dir, dataset, action, img_size=608, batch_size=4, shuffle=True, augment=True, mosaic=True, multiscale=True):
    class_names = load_class_names(os.path.join(data_dir, "class.names"))
    data_dir = os.path.join(data_dir, action)

    if dataset == "UCAS_AOD":
        dataset = UCASAODDataset(data_dir, class_names, img_size=img_size, augment=augment, mosaic=mosaic, multiscale=multiscale)
    
    elif dataset == "DOTA":
        dataset = DOTADataset(data_dir, class_names, img_size=img_size, augment=augment, mosaic=mosaic, multiscale=multiscale)

    elif dataset == "custom":
        dataset = CustomDataset(data_dir, img_size=img_size, augment=augment, mosaic=mosaic, multiscale=multiscale)
    
    else: 
        raise NotImplementedError

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True, collate_fn=dataset.collate_fn)

    return dataset, dataloader
