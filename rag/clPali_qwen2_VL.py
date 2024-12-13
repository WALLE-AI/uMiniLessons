import loguru
from pdf2image import convert_from_path

'''
https://github.com/merveenoyan/smol-vision/blob/main/ColPali_%2B_Qwen2_VL.ipynb
https://github.com/illuin-tech/colpali

'''

from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

def preprocess_files_to_images(file_path):
    images = convert_from_path(file_path)
    return images

def init_model():
    colpai_model = RAGMultiModalModel.from_pretrained("vidore/colpali")
    vlm_model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",
                                                        trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    return colpai_model,vlm_model,processor

def build_index(colpai_model,files_path):
    colpai_model.index(
    input_path=files_path,
    index_name="image_index", # index will be saved at index_root/index_name/
    store_collection_with_index=False,
    overwrite=True
    )
    
def execute_model(vlm_model,processor,images,text_query,recall_results):
    image_index = recall_results[0]["page_num"] - 1
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": images[image_index],
                },
                {"type": "text", "text": text_query},
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
    inputs = inputs.to("cuda")

    generated_ids = vlm_model.generate(**inputs, max_new_tokens=50)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    loguru.logger.info(f"response:{output_text}")
    
    
def multi_rag():
    file_path = ""
    images = preprocess_files_to_images(file_path)
    colpai_model,vlm_model,processor = init_model()
    ##构建索引
    build_index(colpai_model=colpai_model,files_path=file_path)
    text_query = "How much did the world temperature change so far?"
    recall_results = colpai_model.search(text_query, k=1)
    execute_model(vlm_model,processor,images,text_query,recall_results)
    
if __name__ == "__main__":
    loguru.logger.info(f"colpali rag start")
    
    
    
    
    

