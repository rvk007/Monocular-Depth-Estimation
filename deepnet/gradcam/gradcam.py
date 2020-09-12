import torch
import torch.nn.functional as F

class GradCAM:
    """Calculate GradCAM saliency map.
    Arguments:
        model: Model network
        layer_name: Layer of the model on which GradCAM will produce saliency map
        input: Input image
    """
    def __init__(self, model, layer_name, input):

        self.model = model
        self.layer_name = layer_name

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]

        def forward_hook(module, input, output):
            self.activations['value'] = output

        self.find_resnet_layer()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
        
        self.saliency_map(input)

    def find_resnet_layer(self):
        """Assign the target layer of the model"""
        layer_num = int(self.layer_name.lstrip('layer'))
        if layer_num == 1:
            self.target_layer = self.model.layer1
        elif layer_num == 2:
            self.target_layer = self.model.layer2
        elif layer_num == 3:
            self.target_layer = self.model.layer3
        elif layer_num == 4:
            self.target_layer = self.model.layer4


    def saliency_map(self, input, class_idx=None, retain_graph=False):
        """Creates saliency map of the same spatial dimension with input
        Arguments:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        """
        b, c, h, w = input.size()

        logit = self.model(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        self.saliency_map =  saliency_map
    
    @property
    def result(self):
        """Returns saliency map"""
        return self.saliency_map

    
