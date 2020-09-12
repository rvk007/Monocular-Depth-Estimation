import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

class Tensorboard:
    """ Creates Tensorboard visualization """
    def __init__(self, logdir, images):
        self.tb = SummaryWriter(logdir)
        self.images = images

    def tbimages(self, model, name, epoch):
        """Add image data to summary
        Arguments:
            model: Network model
            name(string or list): Name of the image
            epoch(integer): Current epoch
        """
        images = model(self.images)

        if isinstance(images, tuple):
            for idx,img in enumerate(images):
                image = torch.sigmoid(img).detach().cpu()
                img_grid = torchvision.utils.make_grid(image)
                self.tb.add_image(f'img_{idx}_{name}_{epoch}.png', img_grid)

                torchvision.utils.save_image(image,f'img_{idx}_{name}_{epoch}.png')
        else:
            images = torch.sigmoid(images)
            images = images.detach().cpu()

            img_grid = torchvision.utils.make_grid(images)
            self.tb.add_image(f'img_{name}_{epoch}.png', img_grid)

            torchvision.utils.save_image(images,f'img_{name}_{epoch}.png')

    def tbmodel(self, model):
        """Add model to summary
        Arguments:
            model: Network model
            model_input: 
        """
        self.tb.add_graph(model, self.images)

    def tbmetrics(self,name, data, x_label):
        """
        Add scalar data to summary
        Arguments:
            name(string or list): Name of the metric
            value(float or list): Metric value
        """
        if isinstance(name, str) and isinstance(data, float):
            self.tb.add_scalar(name, data, x_label)

        elif isinstance(name, str) and isinstance(data, dict):
            for metric_name, metric_tensor in data.items():
                self.tb.add_scalar(metric_name, metric_tensor, x_label)
                


