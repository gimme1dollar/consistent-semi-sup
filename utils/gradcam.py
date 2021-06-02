import torch
from torch.autograd import grad
import torch.nn.functional as F
import sys
class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = torch.squeeze(output)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = torch.squeeze(grad_output[0])
    
    def clear_list(self):
        del self.gradients
        torch.cuda.empty_cache()
        self.gradients = None
    
    def buffer_clear(self):
        del self.gradients, self.activations
        torch.cuda.empty_cache()
        self.gradients = None
        self.activations = None

    def __call__(self, x):
        self.gradients = None
        self.activations = None       
        return self.model(x)

class GradCAM:
    def __init__(self, 
                 model, 
                 target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self.target_layer = target_layer
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = torch.squeeze(output)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = torch.squeeze(grad_output[0])
    
    def clear_list(self):
        del self.gradients
        torch.cuda.empty_cache()
        self.gradients = None
    
    def buffer_clear(self):
        del self.gradients, self.activations
        torch.cuda.empty_cache()
        self.gradients = None
        self.activations = None
    
    def forward(self, input_img):
        return self.model(input_img)
    
    def get_cam(self, input_tensor, label):
        # input_tensor : b x c x h x w
        self.buffer_clear()
        self.model.eval()
        
        cam_stack=[]    
        output = self.model(input_tensor) # 1 x c x h x w

        for batch_idx in range(input_tensor.shape[0]): # batch 개
            self.model.zero_grad()

            y_c = output[batch_idx, label[batch_idx]]
            y_c.backward(retain_graph=True)     

            activations = self.activations[batch_idx]
            grads = self.gradients[batch_idx]

            weights = torch.mean(grads, dim=(1, 2), keepdim=True)
            cam = torch.sum(weights * activations, dim=0)
            cam = cam.unsqueeze(0).unsqueeze(0)

            min_v = torch.min(cam)
            range_v = torch.max(cam) - min_v

            if range_v > 0:
                cam = (cam - min_v) / range_v
            else:
                cam = torch.zeros(cam.size())

            cam_stack.append(cam)
            self.clear_list()
            del y_c, activations, grads, weights, cam
            torch.cuda.empty_cache()

        concated_cam = torch.cat(cam_stack, dim=0).squeeze() # b x h x w
        del cam_stack, input_tensor, output
        torch.cuda.empty_cache()
        self.model.train()
        self.buffer_clear()
        return concated_cam
