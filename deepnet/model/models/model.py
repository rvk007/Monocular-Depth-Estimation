from deepnet.model.models.resnet import ResNet, BasicBlock
from deepnet.model.models.customnet import CustomNet, CustomBlock
from deepnet.model.models.resmodnet import ResModNet, ModBasicBlock
from deepnet.model.models.masknet import MaskNet3
from deepnet.model.models.depthnet import DepthMaskNet8

def ResNet18():
    """Create Resnet-18 architecture
    Returns:
        Resnet-18 architecture
    """
    return ResNet(BasicBlock, [2,2,2,2])

def CustomRes():
    """Create CustomNet architecture
    Returns:
        CustomNet architecture
    """
    return CustomNet(CustomBlock)

def ResModNet18():
    """Create ResModnet-18 architecture
    Returns:
        ResModnet-18 architecture
    """
    return ResModNet(ModBasicBlock, [2,2,2,2])

def MaskNet():
    """Create MaskNet-3 architecture
    Returns:
        MaskNet-3 architecture
    """
    return MaskNet3()

def DepthMaskNet():
    """Create DepthMaskNet-8 architecture
    Returns:
        DepthMaskNet-8 architecture
    """
    return DepthMaskNet8()




   