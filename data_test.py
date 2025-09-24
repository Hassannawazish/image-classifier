from torchvision import datasets, transforms
from torch.utils.data import DataLoader


train_dataset_path = "../data/version2/train"
# test_dataset_path = "../data/version2/test"

train_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_dataset = datasets.ImageFolder(train_dataset_path, train_transforms)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True) # Shuffle for randomizing

for data in train_loader:
    print("train dataset")
    images, labels = data
    print(("batch of image shape", images.shape))
    print("labels", labels)
    break
