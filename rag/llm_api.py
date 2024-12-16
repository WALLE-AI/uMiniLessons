
import json
import time
from typing import Generator
import loguru
import openai
import os
import threading
import httpx
from pydantic import BaseModel
import requests

from rag.encoder import jsonable_encoder

class ModelResponseEntity(BaseModel):
    prompt: str = "你是一个有用的助手"
    model_name: str = "dsdsad"
    content: str = "shdhshadhsahd"
    total_tokens: int = 122343

    def to_dict(self) -> dict:
        return jsonable_encoder(self)

MODEL_NAME_LIST = {
    "starvlm":{
        "Qwen2-VL-2B-Instruct":"Qwen2-VL-2B-Instruct"
        
    }
    
}

class LLMApi():
    def __init__(self) -> None:
        self.des = "llm api service"
    def __str__(self) -> str:
        return self.des
    
    def init_client_config(self,llm_type):
        if llm_type == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
            api_key  = os.environ.get("OPENROUTER_API_KEY")
        elif llm_type =='siliconflow':
            base_url = "https://api.siliconflow.cn/v1"
            api_key = os.environ.get("SILICONFLOW_API_KEY")
        elif llm_type=="openai":
            base_url = "https://api.openai.com/v1"
            api_key  = os.environ.get("OPENAI_API_KEY")
        elif llm_type == "starvlm":
            base_url = os.getenv("VLM_SERVE_HOST")
            api_key  = "empty"
        elif llm_type == "starchat":
            base_url = os.getenv("CHAT_SERVE_HOST")
            api_key  = "empty"
            
        return base_url,api_key     
    @classmethod
    def llm_client(cls,llm_type):
        base_url,api_key = cls().init_client_config(llm_type)
        thread_local = threading.local()
        try:
            return thread_local.client
        except AttributeError:
            thread_local.client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key,
                # We will set the connect timeout to be 10 seconds, and read/write
                # timeout to be 120 seconds, in case the inference server is
                # overloaded.
                timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10),
            )
            return thread_local.client
    @classmethod
    def build_image_prompt(cls,query,image_base64):
        user_content = [
                {"type": "text",
                 "text": query},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_base64,
                    },
                }
        ]
        prompt = [{"role": "user", "content": user_content}]
        return prompt
    
    @classmethod
    def build_prompt(cls,query,system_prompt=None,search=False):
        #这里可以加上ddsg api接口
        if system_prompt:
            prompt = [{"role":"system","content":system_prompt},
                  {"role": "user", "content": query}]
        else:
            prompt = [{"role": "user", "content": query}]
        return prompt
    
    @classmethod
    def messages_stream_generator(cls,response):
        message_content = ""
        for text in response:
            ##finsh_reason获取usage内容
            if text.choices[0].finish_reason == "stop":
                usage_info_dict={}
                if text.usage:
                    usage_info_dict = text.usage.to_dict()
                else:
                    ##total tokens
                    usage_info_dict['total_tokens'] = cls._get_num_tokens_by_gpt2(message_content)
                if text.choices[0].delta.content:
                    message_content += text.choices[0].delta.content
                response_dict = ModelResponseEntity(
                model_name=text.model,
                content=message_content,
                total_tokens=usage_info_dict['total_tokens']
                )
                return response_dict.to_dict()
            else:
                if text.choices[0].delta.content:
                    message_content += text.choices[0].delta.content
        ##如果没有就返回为默认
        response_dict = ModelResponseEntity(
            model_name=text.model,
            content=message_content,
            total_tokens=cls._get_num_tokens_by_gpt2(message_content)
        )
        return response_dict.to_dict()
    
    @classmethod
    def _get_num_tokens_by_gpt2(self, text: str) -> int:
        """
        Get number of tokens for given prompt messages by gpt2
        Some provider models do not provide an interface for obtaining the number of tokens.
        Here, the gpt2 tokenizer is used to calculate the number of tokens.
        This method can be executed offline, and the gpt2 tokenizer has been cached in the project.

        :param text: plain text of prompt. You need to convert the original message to plain text
        :return: number of tokens 实际tokens计算有点误差
        """
        return 0

    
    @classmethod
    def call_llm_no_postprocess(cls,prompt,stream=True,llm_type="siliconflow",model_name="Qwen/Qwen2.5-72B-Instruct",response_format=None):
        '''
        默认选择siliconflow qwen2-72B的模型来
        '''
        llm_response = cls.get_client(llm_type=llm_type).chat.completions.create(
                model=MODEL_NAME_LIST[llm_type][model_name],
                messages=prompt,
                max_tokens=4096,
                stream=stream,
                temperature=0.2
            )
        return llm_response
    
    
    @classmethod
    def call_llm(cls,prompt,stream=True,llm_type="siliconflow",model_name="Qwen/Qwen2.5-72B-Instruct",response_format=None):
        '''
        默认选择siliconflow qwen2-72B的模型来
        '''
        llm_response = cls.get_client(llm_type=llm_type).chat.completions.create(
                model=MODEL_NAME_LIST[llm_type][model_name],
                messages=prompt,
                max_tokens=4096,
                stream=stream,
                temperature=0.2
            )
        if stream:
            return cls.messages_stream_generator(llm_response)
        else:
            response_dict = ModelResponseEntity(
                model_name=llm_response.model,
                content=llm_response.choices[0].message.content,
                total_tokens=llm_response.usage.total_tokens
            )
            return response_dict.to_dict()
    

    @classmethod    
    def get_client(cls,llm_type):
        return cls().llm_client(llm_type)
