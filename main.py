import json

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from skimage.draw import polygon

from libs.CityScapesDataset import CityscapesDataset
from libs.Extract_Label_ID_Maps import Label_Id_Maps_Extracter
from libs.model import Semantic_Segmentater
from libs.train_model import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = Semantic_Segmentater()
    train(model=model, epochs=30, lr=0.01, data_dir=r'D:\python\SemanticSegmentation_CityScapes\dataset')