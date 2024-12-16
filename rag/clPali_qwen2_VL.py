import base64
from io import BytesIO
from pathlib import Path
import loguru
from pdf2image import convert_from_path

from rag.llm_api import LLMApi

'''
https://github.com/merveenoyan/smol-vision/blob/main/ColPali_%2B_Qwen2_VL.ipynb
https://github.com/illuin-tech/colpali
##https://github.com/openbmb/visrag

'''
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

vlm_models_path = "/home/dataset1/gaojing/models/Qwen2-VL-2B-Instruct"
colpali_model_path = "/home/dataset1/gaojing/models/colpali-v1.3"


def preprocess_files_to_images(file_path):
    images = convert_from_path(file_path)
    return images

def init_model():
    ##40G显卡会超出问题
    # vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(vlm_models_path,
    #                                                     trust_remote_code=True, torch_dtype=torch.bfloat16).cuda(device=0)
    # processor = AutoProcessor.from_pretrained(vlm_models_path, trust_remote_code=True)
    colpai_model = RAGMultiModalModel.from_pretrained(colpali_model_path)
    return colpai_model

def build_index(colpai_model,files_path):
    colpai_model.index(
    input_path=files_path,
    index_name="image_index", # index will be saved at index_root/index_name/
    store_collection_with_index=False,
    overwrite=True
    )
    
def im_2_b64(image):
    buff = BytesIO()
    mime_type = "png"
    image.save(buff, format=mime_type)
    img_base64 = base64.b64encode(buff.getvalue()).decode('utf-8')
    return f'data:image/{mime_type};base64,{img_base64}'
    
def execute_model_api(images,text_query,recall_results,file_name):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_query},
            ],
        }
    ]
    #file_name
    for image in recall_results:
        image_index = image["page_num"]
        loguru.logger.info(f"recall images index:{image_index}")
        save_image = "datasets/images/pdf/"+file_name +"_"+str(image_index)+".png"
        images[image_index].save(save_image)
        image_url = {
            "type": "image_url",
            "image_url": {
                "url": im_2_b64(images[image_index])
            }
        }
        messages[0]['content'].append(image_url)
    llm_type = "starvlm"
    model_name = "Qwen2-VL-7B-Instruct"
    result = LLMApi.call_llm(prompt=messages,llm_type=llm_type,model_name=model_name)
    loguru.logger.info(f"repsonse:{result['content']}")
    
    
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
    inputs = inputs.to("cuda:1")

    generated_ids = vlm_model.generate(**inputs, max_new_tokens=50)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    loguru.logger.info(f"response:{output_text}")
    

def multi_rag():
    file_path = "datasets/pdf/ColPaliEfficient_Document_Retrieval_with_Vision_Language_Models.pdf"
    file_name = Path(file_path).name
    images = preprocess_files_to_images(file_path)
    colpai_model= init_model()
    ##构建索引
    build_index(colpai_model=colpai_model,files_path=file_path)
    text_query = ["What does table1 in ColPaliEfficient_Document_Retrieval_with_Vision_Language_Models mainly talk about "]
    for text in text_query:
        recall_results = colpai_model.search(text, k=2)
        execute_model_api(images,text,recall_results,file_name)
    
if __name__ == "__main__":
    loguru.logger.info(f"colpali rag start")
    
    
    
    
    

