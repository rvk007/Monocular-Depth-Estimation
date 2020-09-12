
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor

class Transformations:
    def __init__(
    self, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), pad_dim=(0,0), random_crop_dim=(0,0), resize=(0,0),
    horizontal_flip=0, vertical_flip=0, rotate_degree=0, rotation=0, cutout=0, cutout_dim=(1,1),
    hsv=0, iso_noise=0, bright_contrast=0, gaussian_blur=0, train=False, modest_input=True):

        """Transformations to be applied on the data
        Arguments:
            mean : Tuple of mean values for each channel
                (default: (0.5,0.5,0.5))
            std : Tuple of standard deviation values for each channel
                (default: (0.5,0.5,0.5))
            pad_dim (tuple, optional): Pad side of the image
                pad_dim[0]: minimal result image height (int)
                pad_dim[1]: minimal result image width (int)
                (default: (0,0))
            random_crop_dim (tuple, optional): Crop a random part of the input
                random_crop_dim[0]: height of the crop (int)
                random_crop_dim[1]: width of the crop (int)
                (default: (0,0))
            resize (tuple, optional): Resize input
                resize[0]: new height of the input (int)
                resize[1]: new width of the input (int)
                (default: (0,0))
            horizontal_flip (float, optional): Probability of image being flipped horizontaly 
                (default: 0)
            vertical_flip (int, optional): Probability of image being flipped vertically 
                (default: 0)
            rotation (int, optional): Probability of image being rotated 
                (default: 0)
            cutout (int, optional): Probability of image being cutout 
                (default: 0)
            cutout_dim (list, optional): Cutout a random part of the image
                cutout_dimtransformations.append(ToTensor())[0]: height of the cutout (int)
                cutout_dim[1]: width of the cutout (int)
                (default: (1,1))
            transform_train : If True, transformations for training data else for testing data
                (default : False)  
        Returns:
            Transformations that is to applied on the data
        """
        
        transformations=[]
        if train:
            if sum(pad_dim)>0:
                transformations.append(A.PadIfNeeded(min_height=pad_dim[0], min_width=pad_dim[1], p=1.0))

            if sum(random_crop_dim)>0:
                transformations.append(A.RandomCrop(height = random_crop_dim[0], width = random_crop_dim[1], p=1.0))

            if horizontal_flip:
                transformations.append(A.HorizontalFlip(p=horizontal_flip))

            if vertical_flip:
                transformations.append(A.VerticalFlip(p=vertical_flip))
            
            if gaussian_blur:
                transformations.append(A.GaussianBlur(p=gaussian_blur))

            if rotation:
                transformations.append(A.Rotate(limit=rotate_degree, p=rotation))

            if cutout:
                transformations.append(A.CoarseDropout(
                    max_holes=1, fill_value=tuple(x*255 for x in mean),
                    max_height=cutout_dim[0],max_width=cutout_dim[1],
                    min_height=1, min_width=1, p=cutout
                    ))

            if hsv:
                transformations.append(A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, 
                    val_shift_limit=20, always_apply=False, p=hsv)
                    )

            if iso_noise:
                transformations.append(A.ISONoise(
                    color_shift=(0.01, 0.05), intensity=(0.1, 0.5), 
                    always_apply=False, p=iso_noise)
                    )

            if bright_contrast:
                transformations.append(A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True,
                    always_apply=False, p=bright_contrast)
                )

        if modest_input:
            transformations.append(A.Normalize(mean=mean, std=std,always_apply=True))

        if sum(resize)>0:
            transformations.append(A.Resize(height=resize[0], width=resize[1], interpolation=1, always_apply=False, p=1))

        transformations.append(ToTensor())

        self.transform= A.Compose(transformations)

    def __call__(self, image):
        """Transform the image through the data transformation pipeline
        Arguments:
            image : Image to be transformed
        Returns:
            Transformed image
        """
        image=np.array(image)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        image=self.transform(image=image)['image']
        
        return image