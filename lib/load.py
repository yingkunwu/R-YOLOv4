import os
import torch

from datasets.custom_dataset import CustomDataset
from datasets.UCASAOD_dataset import UCASAODDataset
from datasets.DOTA_dataset import DOTADataset


def load_data(data_dir, class_names, dataset_type, hyp, img_size=608, batch_size=4, augment=False, shuffle=True):
    if dataset_type == "UCAS_AOD":
        dataset = UCASAODDataset(data_dir, class_names, hyp, img_size=img_size, augment=augment)
    elif dataset == "DOTA":
        dataset = DOTADataset(data_dir, class_names, hyp, img_size=img_size, augment=augment)
    elif dataset == "custom":
        dataset = CustomDataset(data_dir, class_names, hyp, img_size=img_size, augment=augment)
    else:
        raise NotImplementedError

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True, collate_fn=dataset.collate_fn)

    return dataset, dataloader
