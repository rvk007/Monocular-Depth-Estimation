import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


from deepnet.data.cifar10 import CIFAR10
from deepnet.gradcam.gradcam import GradCAM
from deepnet.gradcam.gradcam_pp import GradCAMpp
from deepnet.gradcam.visualize_cam import visualize_cam


class _GradCAM:
    """Process the input image through the model and produce the GradCAM heatmap
    Arguments:
        mean (tuple): Mean values for each channel
        std (tuple): Standard deviation values for each channel
        height (int): Required height
        width (int): Required width
        device (str): Device (GPU/CPU)
    """

    def __init__(self, mean, std, height, width, device):
        self.mean = mean
        self.std = std
        self.height = height
        self.width = width
        self.device = device

    def denormalize(self,tensor):
        """Denormalize the image
        Arguments:
            tensor: Input tensor
        Returns:
            Denormalized tensor
        """

        if not tensor.ndimension() == 4:
            raise TypeError('tensor should be 4D')

        mean = torch.FloatTensor(self.mean).view(1, 3, 1, 1).expand_as(tensor).to(self.device)
        std = torch.FloatTensor(self.std).view(1, 3, 1, 1).expand_as(tensor).to(self.device)

        return tensor.mul(std).add(mean)


    def normalize(self,tensor):
        """Normalize the image
        Arguments:
            tensor: Input tensor
        Returns:
            Normalized tensor
        """

        if not tensor.ndimension() == 4:
            raise TypeError('tensor should be 4D')

        mean = torch.FloatTensor(self.mean).view(1, 3, 1, 1).expand_as(tensor).to(self.device)
        std = torch.FloatTensor(self.std).view(1, 3, 1, 1).expand_as(tensor).to(self.device)

        return tensor.sub(mean).div(std)

    def preprocess_image(self,image):
        """Returns the processed image
        Arguments:
            tensor: Input image
        Returns:
            Processed image
        """

        torch_img = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).unsqueeze(0).float().div(255).to(self.device)
        torch_img = F.interpolate(torch_img, size=(self.height,self.width), mode='bilinear', align_corners=False)
        normalized_image = self.normalize(torch_img)
        self.torch_img = torch_img
        return normalized_image

    def gradcam(self, model, layer_name, input):
        """Process GradCam on the input image layer by layer
        Arguments:
            model: Model network
            layer_name: Layer of the model on which GradCAM will produce saliency map
            input: Input image
        """

        self.layer_name = layer_name

        gradcam_mask = []
        for layer in layer_name:
            gradcam = GradCAM(model, layer, input)
            gradcam_mask.append(gradcam.result)

        self.plot(gradcam_mask)

        gradcam_pp_mask = []
        for layer in layer_name:
            gradcam_pp = GradCAM(model, layer, input)
            gradcam_pp_mask.append(gradcam_pp.result)

        #self.plot(gradcam_pp_mask)
        
    def plot(self, images):
        """Plot the images
        Arguments:
            images: List of saliency maps
        """

        results = [self.torch_img.squeeze().cpu()]
        for map in images:
            heatmap, result = visualize_cam(map.cpu(), self.torch_img)
            results.append(result)

        title = ['Result'] + self.layer_name
        _, axs = plt.subplots(1, 5, figsize=(12, 12))
        axs = axs.flatten()
        for idx, value in enumerate(zip(results, axs)):
            img, ax = value
            ax.imshow(img.permute(1,2,0))
            ax.axis('off')
            ax.set_title(title[idx], y=1)
        plt.show()

        

