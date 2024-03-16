from cog import BasePredictor, Input
import torch

class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(self, check_cuda):
        return "is_available:"+torch.cuda.is_available()
