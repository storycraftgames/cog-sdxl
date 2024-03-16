from cog import BasePredictor, Input
import torch

class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(self) -> str:
        return "is_available:"+torch.cuda.is_available()
