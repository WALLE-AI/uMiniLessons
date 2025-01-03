import os
import loguru

def test_vllm_client():
    from distilabel.llms import vLLM

    # You can pass a custom chat_template to the model
    llm = vLLM(
        model="/home/dataset1/gaojing/models/Qwen2-VL-2B-Instruct",
        # chat_template="[INST] {{ messages[0]\"content\" }}\\n{{ messages[1]\"content\" }}[/INST]",
    )

    llm.load()

    # Call the model
    output = llm.generate_outputs(inputs=[[{"role": "user", "content": "你是谁你能够做什么"}]])
    loguru.logger.info(f"output:{output}") 
    
def test_client_vllm():
    from distilabel.llms import ClientvLLM

    llm = ClientvLLM(
        base_url=os.environ['SERVE_HOST'],
        tokenizer="/home/dataset1/gaojing/models/Qwen2-VL-72B-Instruct"
    )

    llm.load()

    results = llm.generate_outputs(
        inputs=[[{"role": "user", "content": "你是谁你能够做什么"}]],
        temperature=0.7,
        top_p=1.0,
        max_new_tokens=256,
    )
    loguru.logger.info(f"output:{results}") 
    


def test_distlabel_self_instruct():
    from distilabel.steps.tasks import SelfInstruct
    from distilabel.llms.huggingface import InferenceEndpointsLLM
    from distilabel.llms import ClientvLLM
    self_instruct = SelfInstruct(
        llm = ClientvLLM(
            base_url= os.environ['SERVE_HOST'],
            tokenizer="/home/dataset1/gaojing/models/Qwen2-VL-72B-Instruct"
        ),
        num_instructions=5,  # This is the default value
    )

    self_instruct.load()

    result = next(self_instruct.process([{"input": "instruction"}]))
    loguru.logger.info(f"output:{result}") 
    
    
def test_distlabel_evlo_instruction():
    from distilabel.steps.tasks import EvolComplexityGenerator
    from distilabel.llms.huggingface import InferenceEndpointsLLM
    from distilabel.llms import ClientvLLM

    # Consider this as a placeholder for your actual LLM.
    evol_complexity_generator = EvolComplexityGenerator(
        llm = ClientvLLM(
            base_url= os.environ['SERVE_HOST'],
            tokenizer="/home/dataset1/gaojing/models/Qwen2-VL-72B-Instruct"
        ),
        num_instructions=5,  # This is the default value
    )

    evol_complexity_generator.load()

    result1 = next(evol_complexity_generator.process())   
    loguru.logger.info(f"output1:{result1}")
    
    from distilabel.steps.tasks import EvolInstructGenerator
    evol_instruct_generator = EvolInstructGenerator(
        llm = ClientvLLM(
            base_url= os.environ['SERVE_HOST'],
            tokenizer="/home/dataset1/gaojing/models/Qwen2-VL-72B-Instruct"
        ),
        num_instructions=5,  # This is the default value
    )

    evol_instruct_generator.load()

    result2 = next(evol_instruct_generator.process())
    loguru.logger.info(f"output2:{result2}") 
    
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    loguru.logger.info(f"test distlabel")
    test_distlabel_self_instruct()
    