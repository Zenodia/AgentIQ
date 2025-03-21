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

@register_function(config_type=LocalGPTSWLM, framework_wrappers=[LLMFrameworkEnum.HF])
async def local_huggingface_gptsw3_workflow(config: LocalGPTSWLM, builder: Builder):
    import torch
    from colorama import Fore
    tokenizer, model = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.HF)
    if config.use_lang_detect_tool:
        lang_tool = builder.get_tool(fn_name=config.lang_id_tool, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    print(type(tokenizer), type(model), config.device)
    async def predict_sentiment(texts):
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(config.device)
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
        labels=torch.argmax(probabilities, dim=-1).tolist()
        return [sentiment_map[p] for p in labels]

    async def _response_fn(input_message: str) -> str:
        logger.info("input_message=%s", input_message)
        if config.is_auto_regressive: 
            input_ids = tokenizer(input_message, return_tensors="pt")["input_ids"].to(config.device)
            
            generated_token_ids = model.generate(
                inputs=input_ids,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.6,
                top_p=1,
            )[0]
            generated_text = tokenizer.decode(generated_token_ids)
        else:
            output_text=await predict_sentiment([input_message])
            generated_text=f"{input_message} | Sentiment:{output_text[0]}"
        if config.use_lang_detect_tool : 
            lang_id = (await lang_tool.ainvoke(input_message))        
            final_output=f"detected_language : {lang_id} | output text : {generated_text}"
        else:
            final_output=generated_text
        print(Fore.CYAN+"final response :\n", final_output , Fore.RESET)
        return final_output

    yield _response_fn
    