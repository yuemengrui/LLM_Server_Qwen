# *_*coding:utf-8 *_*
# @Author : YueMengRui
import time
import torch
from typing import List
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def torch_gc(device):
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


class LLM:

    def __init__(self, model_name_or_path, model_name, logger=None, device='cuda', **kwargs):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        self.logger = logger
        self._load_model(model_name_or_path)
        self.max_length = self.model.config.max_position_embeddings
        self.max_new_tokens = 2048

        if self.logger:
            self.logger.info(str({'config': self.model.config}) + '\n')
            self.logger.info(str({'max_length': self.max_length, 'max_new_tokens': self.max_new_tokens}) + '\n')

        # warmup
        self.lets_chat('你好', [], stream=False)

    def _load_model(self, model_name_or_path):

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.device = self.model.device

    def check_token_len(self, prompt: str):
        code = True
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        prompt_token_len = self.token_counter(messages)
        if prompt_token_len > self.max_length:
            code = False

        return code, prompt_token_len, self.max_length, self.model_name

    def token_counter(self, messages: List):

        return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    def select_history(self, prompt, history, max_prompt_length):
        pass

    def lets_chat(self, prompt, history=[], stream=True, generation_configs={}, **kwargs):

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        if self.logger:
            self.logger.info(str({'prompt_tokens': len(model_inputs.input_ids[0]), 'prompt_str_len': len(prompt),
                                  'prompt': prompt}) + '\n')

        if stream:
            def stream_generator():
                start = time.time()
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                thread = Thread(target=self.model.generate,
                                kwargs=dict(inputs=model_inputs.input_ids, streamer=streamer,
                                            max_new_tokens=self.max_new_tokens))
                thread.start()

                answer = ''
                for resp in streamer:
                    answer += resp
                    generation_tokens = len(self.tokenizer.encode(answer))
                    time_cost = time.time() - start
                    average_speed = f"{generation_tokens / time_cost:.3f} token/s"
                    torch_gc(self.device)
                    yield {"model_name": self.model_name,
                           "answer": answer,
                           "history": history,
                           "time_cost": {"generation": f"{time_cost:.3f}s"},
                           "usage": {"prompt_tokens": len(model_inputs.input_ids[0]), "generation_tokens": generation_tokens,
                                     "total_tokens": len(model_inputs.input_ids[0]) + generation_tokens, "average_speed": average_speed}}

            return stream_generator()

        else:
            start = time.time()
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=self.max_new_tokens
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            resp = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            generation_tokens = len(generated_ids[0])
            time_cost = time.time() - start
            average_speed = f"{generation_tokens / time_cost:.3f} token/s"

            torch_gc(self.device)

            return {
                "model_name": self.model_name,
                "answer": resp,
                "history": history,
                "time_cost": {"generation": f"{time_cost:.3f}s"},
                "usage": {
                    "prompt_tokens": len(model_inputs.input_ids[0]),
                    "generation_tokens": generation_tokens,
                    "total_tokens": len(model_inputs.input_ids[0]) + generation_tokens,
                    "average_speed": average_speed
                }
            }
