from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

colpali_model_path = "/home/dataset1/gaojing/models/colpali-v1.3"
vlm_models_path = "/home/dataset1/gaojing/models/Qwen2-VL-2B-Instruct"
device = get_torch_device("auto")
import loguru
import torch
from qwen_vl_utils import process_vision_info

def init_model():
    model = ColPali.from_pretrained(
        colpali_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    # Load the processor
    processor = ColPaliProcessor.from_pretrained(colpali_model_path)
    return model,processor

def init_qwen2vl_model():
    vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(vlm_models_path,
                                                        trust_remote_code=True, torch_dtype=torch.bfloat16).cuda(device=1)
    processor = AutoProcessor.from_pretrained(vlm_models_path, trust_remote_code=True)
    return vlm_model,processor
    

def model_inference(model,processor):
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/home/dataset1/gaojing/llm/uMiniLessons/datasets/images/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda:1")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    
def execute_model():
    vlm_model,processor=init_qwen2vl_model()
    model_inference(vlm_model,processor)
    

    
if __name__ == "__main__":
    loguru.logger.info(f"colpali rag start")
    init_model()