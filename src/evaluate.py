import torch 
import torch.nn as nn
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from dataset_loader import load_realwaste
from cnn_model import SimpleCNN
from datetime import datetime
import os

#Identifying the device type
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating Device: {device}")

#loading the testing dataset
_,_,test_loader,classes=load_realwaste("data/RealWaste",batch_size=32)

#loading the trained model
model=SimpleCNN(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load("results/models/realwaste_cnn_best_2.pth",map_location=device))
model.eval()
print("Model loaded Successfully")

all_preds=[]
all_labels=[]

with torch.no_grad():
    correct=0
    total=0
    for images,labels in test_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
test_accuracy=100*(correct/total)       
print(f"Accuracy is:{test_accuracy:.2f}%")


#classification metrics
os.makedirs("report", exist_ok=True)
print("\n Classification matircs are as below")
report=classification_report(all_labels,all_preds,target_names=classes,digits=3)
print("\n--- Classification Report ---")
print(report)
#saving the metrics in results folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = f"reports/classification_report_{timestamp}.txt"
with open(report_path, "w") as f:
    f.write(f"Test Accuracy: {test_accuracy:.2f}%\n\n")
    f.write("--- Classification Report ---\n")
    f.write(report)

print(f"\n Classification report saved to {report_path}")

#confusion matrix
cm=confusion_matrix(all_labels,all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - RealWaste Test Set")
plt.tight_layout()
plt.savefig("results/plots/confusion_matrix.png")
plt.show()

print("Confusion Matrix saved at resuls/plots/confusion_matrix.png")



