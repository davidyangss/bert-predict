# After run_classification.py, I got Fine-Tuning bert model for classification. these files locate at onnx-model/google-bert-chinese/hfoptimum-trained.
# Will use onnx-model/google-bert-chinese/hfoptimum-trained to inference some sentences


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
tokenizer = AutoTokenizer.from_pretrained("onnx-model/google-bert-chinese/hfoptimum-trained")
model = AutoModelForSequenceClassification.from_pretrained("onnx-model/google-bert-chinese/hfoptimum-trained")
text = "怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片！开始还怀疑是不是赠送的个别现象，可是后来发现每张DVD后面都有！真不知道生产商怎么想的，我想看的是猫和老鼠，不是米老鼠！如果厂家是想赠送的话，那就全套米老鼠和唐老鸭都赠送，只在每张DVD后面添加一集算什么？？简直是画蛇添足！！"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted_class_id = logits.argmax().item()
print(f"model.config = {model.config}")
print(f"logits = {logits}")
print(f"inputs = {inputs}, predicted = {predicted_class_id}")

