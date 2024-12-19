import paddle
from PIL import Image
from paddlemix.models.aria.modeling_aria import AriaPretrainedModel, AriaForConditionalGeneration
from paddlemix.processors.processing_aria import AriaProcessor
from paddlemix.processors.aria_vision_processor import AriaVisionProcessor
import json


model_id_or_path = 'rhymes-ai/Aria'

config_path = f"{model_id_or_path}/config.json"
with open(config_path, 'r') as f:
    config = json.load(f)
    print("Config loaded successfully:")
    print(json.dumps(config, indent=2))

try:
    model = AriaForConditionalGeneration.from_pretrained(model_id_or_path)
    print(11)
    processor = AriaProcessor.from_pretrained(model_id_or_path,
        trust_remote_code=True)
    print(12)
    # image = Image.open(requests.get(image_path, stream=True).raw)
    image = Image.open('paddlemix/demo_images/examples_image1.jpg').convert('RGB')
    print(13)
    messages = [{'role': 'user', 'content': [{'text': None, 'type': 'image'}, {
        'text': 'what is the image?', 'type': 'text'}]}]
    # text = processor.apply_chat_template(messages, add_generation_prompt=True)
    text = processor.apply_chat_template(messages)
    inputs = processor(text=text, images=image, return_tensors='pd')
    inputs['pixel_values'] = inputs['pixel_values'].to(model.dtype)
    inputs = {k: v.to(model.place) for k, v in inputs.items()}
    with paddle.no_grad(), paddle.amp.auto_cast(dtype='bfloat16'):
        output = model.generate(**inputs, max_new_tokens=500, stop_strings=[
            '<|im_end|>'], tokenizer=processor.tokenizer, do_sample=True,
            temperature=0.9)
        output_ids = output[0][tuple(inputs['input_ids'].shape)[1]:]
        result = processor.decode(output_ids, skip_special_tokens=True)
    print(result)
except Exception as e:
    print(f"Error loading model: {e}")