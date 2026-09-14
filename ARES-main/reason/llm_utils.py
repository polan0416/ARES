import os
import time
import inspect
import logging
from functools import partial

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

from prompts import icl_user_prompt, icl_ass_prompt


class _OpenAIRateLimitError(Exception):
    pass


OPENAI_RATE_LIMIT_ERROR = openai.RateLimitError if openai is not None else _OpenAIRateLimitError


def llm_init(
    model_name,
    tensor_parallel_size=1,
    max_seq_len_to_capture=8192,
    max_tokens=4000,
    seed=0,
    temperature=0,
    frequency_penalty=0,
    num_workers=16,
):
    if "gpt" not in model_name:
        if LLM is None or SamplingParams is None:
            raise ImportError("vllm is required for local model inference but is not installed in the current environment.")
        try:
            logging.getLogger("vllm").setLevel(logging.WARNING)
        except Exception:
            pass

        llm_kwargs = {"model": model_name, "tensor_parallel_size": tensor_parallel_size}
        try:
            sig = inspect.signature(LLM.__init__)
            params = sig.parameters
            if "max_seq_len_to_capture" in params:
                llm_kwargs["max_seq_len_to_capture"] = max_seq_len_to_capture
            if "chat_template_content_format" in params:
                llm_kwargs["chat_template_content_format"] = "string"
            if num_workers > 1 and "max_num_seqs" in params:
                llm_kwargs["max_num_seqs"] = num_workers
        except (TypeError, ValueError):
            pass
        if num_workers > 1 and "max_num_seqs" not in llm_kwargs:
            llm_kwargs["max_num_seqs"] = num_workers
        client = LLM(**llm_kwargs)
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                         frequency_penalty=frequency_penalty)
        llm = partial(client.chat, sampling_params=sampling_params, use_tqdm=False)
    else:
        if OpenAI is None:
            raise ImportError("openai is required for GPT inference but is not installed in the current environment.")
        client = OpenAI()
        llm = partial(client.chat.completions.create, model=model_name, seed=seed, temperature=temperature, max_tokens=max_tokens)
    return llm


def get_outputs(outputs, model_name):
    if "gpt" not in model_name:
        return outputs[0].outputs[0].text
    else:
        return outputs.choices[0].message.content


def get_output_text(output, model_name):
    if "gpt" not in model_name:
        return output.outputs[0].text
    return output.choices[0].message.content


def build_conversation(prompts, mode):
    conversation = []
    if "sys" in mode:
        conversation.append({"role": "system", "content": prompts["sys_query"]})
    if "icl" in mode:
        conversation.append({"role": "user", "content": icl_user_prompt})
        conversation.append({"role": "assistant", "content": icl_ass_prompt})
    if "sys" in mode:
        conversation.append({"role": "user", "content": prompts["user_query"]})
    return conversation


def needs_dc_followup(first_answer):
    low = first_answer.lower()
    return (
        "ans:" not in low
        or "ans: not available" in low
        or "ans: no information available" in low
    )


def llm_inf(llm, prompts, mode, model_name):
    res = []
    if 'sys' in mode:
        conversation = [{"role": "system", "content": prompts['sys_query']}]

    if 'icl' in mode:
        conversation.append({"role": "user", "content": icl_user_prompt})
        conversation.append({"role": "assistant", "content": icl_ass_prompt})

    if 'sys' in mode:
        conversation.append({"role": "user", "content": prompts['user_query']})
        outputs = get_outputs(llm(messages=conversation), model_name)
        res.append(outputs)

    if 'sys_cot' in mode:
        if 'clear' in mode:
            conversation = []
        conversation.append({"role": "assistant", "content": outputs})
        conversation.append({"role": "user", "content": prompts['cot_query']})
        outputs = get_outputs(llm(messages=conversation), model_name)
        res.append(outputs)
    elif "dc" in mode:
        if 'ans:' not in res[0].lower() or "ans: not available" in res[0].lower() or "ans: no information available" in res[0].lower():
            conversation.append({"role": "user", "content": prompts['cot_query']})
            outputs = get_outputs(llm(messages=conversation), model_name)
            res[0] = outputs
        res.append("")
    else:
        res.append("")

    return res


def llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries):
    retries = 0
    while retries < max_retries:
        try:
            return llm_inf(llm, each_qa, llm_mode, model_name)
        except OPENAI_RATE_LIMIT_ERROR:
            wait_time = (2 ** retries) * 5  # Exponential backoff
            print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    raise Exception("Max retries exceeded. Please check your rate limits or try again later.")


def llm_inf_all(llm, each_qa, llm_mode, model_name, max_retries=5):
    if 'gpt' in model_name:
        return llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries)
    else:
        return llm_inf(llm, each_qa, llm_mode, model_name)


def llm_inf_batch(llm, batch_prompts, llm_mode, model_name):
    if not batch_prompts:
        return []
    if "gpt" in model_name:
        return [llm_inf_with_retry(llm, p, llm_mode, model_name, 5) for p in batch_prompts]

    conversations = [build_conversation(p, llm_mode) for p in batch_prompts]
    batch_outputs = llm(messages=conversations)
    results = [[get_output_text(out, model_name)] for out in batch_outputs]

    if "sys_cot" in llm_mode:
        cot_conversations = []
        cot_indices = []
        for idx, (conv, res) in enumerate(zip(conversations, results)):
            cot_conv = list(conv)
            if "clear" in llm_mode:
                cot_conv = []
            cot_conv.append({"role": "assistant", "content": res[0]})
            cot_conv.append({"role": "user", "content": batch_prompts[idx]["cot_query"]})
            cot_conversations.append(cot_conv)
            cot_indices.append(idx)
        cot_outputs = llm(messages=cot_conversations)
        for idx, out in zip(cot_indices, cot_outputs):
            results[idx].append(get_output_text(out, model_name))
    elif "dc" in llm_mode:
        dc_conversations = []
        dc_indices = []
        for idx, (conv, res) in enumerate(zip(conversations, results)):
            if not needs_dc_followup(res[0]):
                res.append("")
                continue
            dc_conv = list(conv)
            dc_conv.append({"role": "assistant", "content": res[0]})
            dc_conv.append({"role": "user", "content": batch_prompts[idx]["cot_query"]})
            dc_conversations.append(dc_conv)
            dc_indices.append(idx)
        if dc_conversations:
            dc_outputs = llm(messages=dc_conversations)
            for idx, out in zip(dc_indices, dc_outputs):
                results[idx][0] = get_output_text(out, model_name)
        for idx in dc_indices:
            results[idx].append("")
    else:
        for res in results:
            res.append("")
    return results


def llm_inf_all_batch(llm, batch_qa, llm_mode, model_name, max_retries=5):
    if "gpt" in model_name:
        return [
            llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries)
            for each_qa in batch_qa
        ]
    return llm_inf_batch(llm, batch_qa, llm_mode, model_name)


def build_chat_messages(system_prompt, user_prompt, example_user=None, example_assistant=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if example_user is not None and example_assistant is not None:
        messages.append({"role": "user", "content": example_user})
        messages.append({"role": "assistant", "content": example_assistant})

    messages.append({"role": "user", "content": user_prompt})
    return messages


def llm_chat_once(llm, system_prompt, user_prompt, model_name, example_user=None, example_assistant=None):
    messages = build_chat_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        example_user=example_user,
        example_assistant=example_assistant,
    )
    return get_outputs(llm(messages=messages), model_name)


def llm_chat_with_retry(
    llm,
    system_prompt,
    user_prompt,
    model_name,
    max_retries=5,
    example_user=None,
    example_assistant=None,
):
    if 'gpt' not in model_name:
        return llm_chat_once(
            llm,
            system_prompt,
            user_prompt,
            model_name,
            example_user=example_user,
            example_assistant=example_assistant,
        )

    retries = 0
    while retries < max_retries:
        try:
            return llm_chat_once(
                llm,
                system_prompt,
                user_prompt,
                model_name,
                example_user=example_user,
                example_assistant=example_assistant,
            )
        except OPENAI_RATE_LIMIT_ERROR:
            wait_time = (2 ** retries) * 5
            print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    raise Exception("Max retries exceeded. Please check your rate limits or try again later.")
