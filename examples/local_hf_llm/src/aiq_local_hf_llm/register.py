import logging

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig


logger = logging.getLogger(__name__)


class LocalGPTSWLM(FunctionBaseConfig, name="local_hf_llm"):
    # Add your custom configuration parameters here
    llm_name: LLMRef = "huggingface_llm"


@register_function(config_type=LocalGPTSWLM, framework_wrappers=[LLMFrameworkEnum.HUGGINGFACE])
async def local_huggingface_gptsw3_workflow(config: LocalGPTSWLM, builder: Builder):
    tokenizer, model = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.HUGGINGFACE)
    print(type(tokenizer), type(model))
    async def _response_fn(input_message: str) -> str:
        logger.info("input_message=%s", input_message)
        input_ids = tokenizer(input_message, return_tensors="pt")["input_ids"].to(device)

        generated_token_ids = model.generate(
            inputs=input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.6,
            top_p=1,
        )[0]

        generated_text = tokenizer.decode(generated_token_ids)


        return generated_text

    yield _response_fn
    