from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Model():
    def __init__(self, model_name, DEFAULT_SYSTEM_PROMPT):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
        self.dsp = DEFAULT_SYSTEM_PROMPT

    def _get_summary(self, query:str):
        prompt = self.tokenizer.apply_chat_template([{
            "role": "system",
            "content": self.dsp
        }, {
            "role": "user",
            "content": query
        }], tokenize=False, add_generation_prompt=True)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        data = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        data = {k: v.to(self.model.device) for k, v in data.items()}
        output_ids = self.model.generate(**data, max_new_tokens=400,eos_token_id=terminators)[0]
        output_ids = output_ids[len(data["input_ids"][0]):]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return output 
    
    def __call__(self, query:str):
        return self._get_summary(query)