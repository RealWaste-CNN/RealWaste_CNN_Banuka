import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset_loader import load_realwaste
from cnn_model import SimpleCNN

#device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

#Defining Hyperparameters
num_classes=9
num_epochs=20
batch_size=32
learning_rate=0.01 #alpha in our notes

#now I will load the dataset
train_loader,val_loader,test_loader,classes=load_realwaste("data/RealWaste",batch_size=batch_size)

#Initializing the model, loss function and tye optimizer
model=SimpleCNN(num_classes=num_classes).to(device)
criterian=nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)#change LR in epochs

#training loop
train_losses=[]
val_losses=[]
train_accuracies=[]
val_accuracies=[]

for epoch in range(num_epochs):
    model.train()
    running_loss=0.0
    correct=0
    total=0

    for images,labels in tqdm(train_loader,desc=f"Epoch[{epoch+1}/{num_epochs}]"):
        images,labels=images.to(device),labels.to(device)

        #Forward Pass
        outputs=model(images)
        loss=criterian(outputs,labels)

        #Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Loss and Accuracy
        running_loss+=loss.item()
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    train_loss=running_loss/len(train_loader)
    train_accuracy=100*correct/total

    #Validation
    model.eval()
    val_loss=0.0
    val_correct=0
    val_total=0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterian(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total

    #logging
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
          f"| Train Acc: {train_accuracy:.2f}% | Val Acc: {val_acc:.2f}%")

#plot these values for visualization purposes
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('results/plots/loss_curve_3.png')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig('results/plots/accuracy_curve_3.png')
plt.show()

#saving the model
torch.save(model.state_dict(), 'results/models/realwaste_cnn_best.pth')
print("Model saved successfully at 'results/models/realwaste_cnn_best.pth'")

