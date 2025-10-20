import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def load_realwaste(data_dir="data/RealWaste", batch_size=32, image_size=(64, 64)): #adjust the image_size, I used 64x64 since I dont have a GPU(save computing power) 
    """
    Loads and splits the RealWaste dataset into train, validation, and test sets.
    Expects directory structure:
        data/realwaste/class_name/image.jpg
    """
    

    # Image preprocessing (resize, convert to tensor, normalize)
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Validation/Test: No augmentation, only normalization
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #Load dataset 
    dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    # Split dataset sizes
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    #Make split reproducible
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=g)

    #Assign proper transforms to val/test sets 
    val_ds.dataset.transform = test_transform
    test_ds.dataset.transform = test_transform

    #Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Print summary
    print(f"Dataset loaded from: {data_dir}")
    print(f"Total images: {total_size}")
    print(f"Train: {train_size} | Validation: {val_size} | Test: {test_size}")
    print(f"Classes: {dataset.classes}")

    return train_loader, val_loader, test_loader, dataset.classes

if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = load_realwaste()
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)