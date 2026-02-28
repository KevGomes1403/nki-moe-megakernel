################################################################################

import pickle
import torch
from collections import OrderedDict
from inspect import signature
from types import MethodType

class ForwardWrapper(object):
    def __init__(self, module_name, forward):
        self.module_name = module_name
        if not isinstance(forward, MethodType):
            raise TypeError('not isinstance(forward, MethodType)')
        self.forward = forward
        self.parameters = list(signature(forward).parameters.keys())
        self.instance = forward.__self__
        self.inputs_output_recorded = False

    def __call__(self, *args, **kwargs):
        if not self.inputs_output_recorded:
            inputs_outputs = OrderedDict()
            for (parameter, arg) in zip(self.parameters, args):
                inputs_outputs[parameter] = pickle.loads(
                    pickle.dumps(arg)
                )
            for (parameter, arg) in kwargs.items():
                inputs_outputs[parameter] = pickle.loads(
                    pickle.dumps(arg)
                )
            result = self.forward(*args, **kwargs)
            inputs_outputs['return'] = pickle.loads(
                pickle.dumps(result)
            )
            torch.save(inputs_outputs, '%s.pt' % (self.module_name,))
            self.inputs_output_recorded = True
            return result
        else:
            return self.forward(*args, **kwargs)

################################################################################

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-30B-A3B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt")

################################################################################
module_types_to_names_and_modules = {}

import random
random.seed(0)

for name, module in model.named_modules():
    module_type = type(module)
    if not module_type.__module__.startswith('torch'):
        module_types_to_names_and_modules.setdefault(module_type, []).append((name, module))

for names_and_modules in module_types_to_names_and_modules.values():
    name, module = random.choice(names_and_modules)
    if not name:
        continue
    module.forward = ForwardWrapper(name, module.forward)
################################################################################

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids)

print("content:", content)