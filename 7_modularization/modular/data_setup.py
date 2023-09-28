"""contrains functionality for creating pytorch dataloader's for image 
    classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS
):
    """Creates a training and testing dataloaders
        turns data into train andtest datasets then dataloaders
    Args:
        train_dir (str): train path 
        test_dir (str): test path
        transforms (transforms.Compose): transformation
        batch_size (int): batch size
        num_workers (int, optional): No. of cpu cores. Defaults to NUM_WORKERS.
    
    Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Example usage:
        train_dataloader, test_dataloader, class_names = create_dataloadres(train_dir = path/to/train_dir,
        test_dir=pathto.test_dir,
        transform=some_transform,
        batch_size=32,
        num_workers=4)
    """
    #soe code

    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                    transform=transform, # transforms to perform on data (images)
                                    target_transform=None) # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir, 
                                    transform=transform)
    #get class names
    class_names = train_data.classes 
    
    #turn image data into dataloaders
    train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=batch_size, # how many samples per batch? 
                              num_workers=num_workers, #no of cpu cores used
                              pin_memory=True,
                              shuffle=True) #keeps memeory in GPU wherever possible

    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=batch_size, 
                                num_workers=num_workers, 
                                shuffle=False,
                                pin_memory=True) # don't usually need to shuffle testing data
    
    
    
    return train_dataloader, test_dataloader, class_names
    
