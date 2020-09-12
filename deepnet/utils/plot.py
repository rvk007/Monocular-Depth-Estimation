import torch
import numpy as np
import matplotlib.pyplot as plt 

class Plot:
    def __init__(self,mean, std, device, model):
        self.mean = mean
        self.std = std
        self.device = device
        self.train_losses, self.train_accuracies, self.test_losses, self.test_accuracies, self.correct_samples, self.incorrect_samples = model.results

    def plot_metric(self, metric):
        """Plot the graph of the given metric
        Arguments:
            values : List of data to be plot
            metric : The metric of data [Loss/Accuracy]
        """
        if metric == 'Accuracy':
            train_values = self.train_accuracies
            test_values = self.test_accuracies
        elif metric == 'Loss':
            train_values = self.train_losses
            test_values = self.test_losses

        # Initialize a figure
        fig = plt.figure(figsize=(7, 5))

        # Plot values
        plt.plot(train_values, 'g', label='Train')
        plt.plot(test_values, 'b', label='Validation')

        # Label axes
        plt.xlabel('Epoch')
        plt.ylabel(metric)

        # Set legend
        location = 'upper right' if metric == 'Loss' else 'lower right'
        plt.legend(loc = location)

        # Save plot
        fig.savefig(f'{metric.lower()}_change.png')
        
        plt.show()
        
    def denormalize(self, tensor):
        """Denormalize the image
        Arguments:
            tensor: Input tensor
        Returns:
            Denormalized tensor
        """
        
        mean = torch.FloatTensor(self.mean).view(3, 1, 1).expand_as(tensor).to(self.device)
        std = torch.FloatTensor(self.std).view(3, 1, 1).expand_as(tensor).to(self.device)
        tensor = tensor.to(self.device).mul(std).add(mean)
        return np.asarray(tensor.cpu().permute(1,2,0))

    def plot_images(self, data, classes, metric):
        """Displays the image prediction
        Arguments:
            data : Validation data of the dataset
            classes : Target Classes of the dataset
            metric : The metric of data [Correct_sample/ Incorrect_sample]
        """
        _, axs = plt.subplots(5, 5, figsize=(12, 12))
        axs = axs.flatten()

        for idx, ax in enumerate(axs):
            sample = data[idx]
            image = self.denormalize(sample['image'])
            label = sample['label'].item()
            prediction = sample['prediction'].item()

            ax.imshow(image)
            ax.axis('off')
            ax.set_title(f'Label: {classes[label]}\nPrediction: {classes[prediction]}', y=1)
            plt.subplots_adjust(top = 0.92, bottom=0.01)

        plt.savefig(metric+'.png')
        plt.show()

    def class_accuracy(self, model, classes, dataloader):
        """Prints Accuracy of each class
        Arguments:
            model : Model Network
            classes: Target classes of the dataset
            dataloader: Test dataloader
        """

        class_total = [0 for i in range(len(classes))]
        class_correct = [0 for i in range(len(classes))]
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                prediction = output.argmax(dim=1, keepdim=False)
                correct = prediction.eq(target)

                for i in range(len(target)):
                    label = target[i]
                    if correct[i]:
                        class_correct[label] += 1
                    class_total[label] += 1

        for i in range(len(class_total)):
            print(f'Accuracy of {classes[i]} : {(class_correct[i]*100)/class_total[i]}% \t Image count: {class_total[i]}')
