from cog import BasePredictor, Input, Path
import os
import base64
import shlex

# Define a list of models, each dictionary contains voice, model path, and speaker ID
models = [
    {"lang": "fr", "speaker": 0, "name": "tom", "model": "./piper/models/fr/tom/fr_FR-tom-medium.onnx"},
    {"lang": "fr", "speaker": 0, "name": "jessica", "model": "./piper/models/fr/upmc/fr_FR-upmc-medium.onnx"},
    {"lang": "fr", "speaker": 1, "name": "pierre", "model": "./piper/models/fr/upmc/fr_FR-upmc-medium.onnx"},
    {"lang": "en", "speaker": 0, "name": "amy", "model": "./piper/models/en/amy/en_US-amy-medium.onnx"},
    {"lang": "en", "speaker": 0, "name": "kristin", "model": "./piper/models/en/kristin/en_US-kristin-medium.onnx"},
    {"lang": "en", "speaker": 0, "name": "kusal", "model": "./piper/models/en/kusal/en_US-kusal-medium.onnx"},
    {"lang": "en", "speaker": 0, "name": "ljspeech", "model": "./piper/models/en/ljspeech/en_US-ljspeech-medium.onnx"}
]

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Setup resources if necessary, omitted for simplicity."""
        pass

    def predict(
        self,
        episode_id: str = Input(description="Episode ID", default=""),
        text: str = Input(description="Text to synthesize"),
        voice: str = Input(description="Voice to use", default="tom"),
    ) -> str:
        safe_text = shlex.quote(text)
        
        #remove the out.wav file if it exists
        if os.path.exists("./out.wav"):
            os.remove("./out.wav")

        # Find the model based on the selected voice
        selected_model = next((m for m in models if m['name'] == voice), None)
        if not selected_model:
            raise ValueError("Selected voice is not available.")

        # Construct the command to run the synthesis process, including the speaker ID
        output_file = "./out.wav"
        model_path = selected_model['model']
        speaker_id = selected_model['speaker']
        command = f"echo {safe_text} | ./piper/piper --model {model_path} --speaker {speaker_id} --output_file {output_file}"
        
        os.system(command)
        
        # Encode the output audio file in base64 create a new file
        with open(output_file, "rb") as audio_file:
            base64_file = base64.b64encode(audio_file.read()).decode("utf-8")
        
        return base64_file
