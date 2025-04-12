import logging

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig
from . import language_detect

logger = logging.getLogger(__name__)


class LocalGPTSWLM(FunctionBaseConfig, name="local_hf_llm"):
    # Add your custom configuration parameters here
    llm_name: LLMRef = "huggingface_llm"
    device: str="cuda:0" # or cpu
    lang_id_tool: FunctionRef
    use_lang_detect_tool: bool
    is_auto_regressive: bool
    sample_img_path: str 

@register_function(config_type=LocalGPTSWLM, framework_wrappers=[LLMFrameworkEnum.HF])
async def local_huggingface_gptsw3_workflow(config: LocalGPTSWLM, builder: Builder):
    from PIL import Image
    from colorama import Fore
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor, model = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.HF)
    #processor = AutoProcessor.from_pretrained(config.llms.huggingface_llm.model_name, trust_remote_code=True)
    print(Fore.RED + "sample_img_path =", config.sample_img_path)
    
    
    image_path = config.sample_img_path
    image = Image.open(image_path)



    async def _response_fn(input_message: str) -> str:
        logger.info("input_message=%s", input_message)
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": input_message}]}
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = processor(images=image, text=text, return_tensors="pt", device_map='auto', padding=True, truncation=True).to('cuda:0')
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(response)

        print(Fore.CYAN+"final response :\n", response , Fore.RESET)
        return response

    yield _response_fn
    