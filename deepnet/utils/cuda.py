import torch

def initialze_cuda(SEED):
    """Initialize the GPU if available
    Arguments: 
        SEED : The value of seed to have amplost same distribution of data everytime we run the model
    Returns:
        cuda: True if GPU is available else False
        device: 'cuda' or 'cpu'
    """
    cuda = torch.cuda.is_available()
    print('Is CUDA Available?', cuda)

    # For reproducibility of results
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    device = torch.device("cuda" if cuda else "cpu")

    return cuda, device



