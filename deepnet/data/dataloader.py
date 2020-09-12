import torch

def data_loader(data, batch_size, num_workers= 2, cuda = False):
    """Loads data on the machine [GPU/CPU]
    Arguments:
        cuda : If True model runs on GPU otherwise on CPU (By default : False)
        train_data : Training Data
        val_data : Validation data
        batch_size : Number of images in a batch
        num_workers : Number of workers simultaneously putting data into RAM (By default: 2)
    Returns:
        Dataloader, after loading data on GPU or CPU
    """
    # dataloader arguments
    dataloader_args ={'shuffle' : True,
                      'batch_size' : batch_size,
                      'num_workers' : num_workers
                      }

    if cuda:
        dataloader_args['pin_memory'] = True

    # train dataloader
    dataloader = torch.utils.data.DataLoader(data, **dataloader_args)

    return dataloader