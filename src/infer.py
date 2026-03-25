
from funasr import AutoModel

model_path = "stage1_outputs"
model = AutoModel(
                model="paraformer-zh", 
                model_path = model_path
                )
res = model.generate(
                input="data/audio.wav", 
                )

print(res[0]["text"])


