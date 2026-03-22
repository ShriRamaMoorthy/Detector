import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import gradio as gr
from PIL import Image
import numpy as np
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features,2)
model.load_state_dict(torch.load('pothgoleguard_resnet18.pth',map_location=device))
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# GRAD-CAM
class GradCAM:
    def __init__(self,model,target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self,module,input,output):
        self.activations = output.detach()

    def save_gradient(self,module,grad_input,grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, *args, **kwds):
        self.model.zero_grad()
        output = self.model(image_tensor)
        pred_class = output.argmax(dim=1).item()
        output[:,pred_class].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=[1,2],keepdim=True)
        cam = (weights * activations).sum(dim=0).relu()
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam,(224,224))
        cam = (cam-cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, pred_class
    
target_layer = model.layer4[-1]
gradcam = GradCAM(model, target_layer)