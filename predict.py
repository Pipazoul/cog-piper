# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os 

models = {
    "fr": [
        {"name": "tom", "model": "./piper/models/fr/tom/fr_FR-tom-medium.onnx"}
    ],
    "en-us": [
        {"name": "rayan", "model": "./piper/models/en_us/rayan/en_US-ryan-high.onnx"}
    ]
}

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        text: str = Input(description="Text to synthesize"),
        lang: str = Input(description="Language of the text", choices=["fr", "en-us"], default="fr"),
        voice: str = Input(description="Voice to use", default="tom"),
    ) -> Path:
        """Run a single prediction on the model"""

        model = models[lang][0]
        output_file = "./out.wav"
        os.system(f"echo '{text}' | ./piper/piper --model {model['model']} --output_file {output_file}")
        return Path(output_file)