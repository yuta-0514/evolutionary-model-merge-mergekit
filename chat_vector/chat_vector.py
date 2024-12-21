import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


if __name__ == "__main__":
    eng_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        token=HF_TOKEN
    )
    eng_tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1"
    )
    math_model = AutoModelForCausalLM.from_pretrained(
        "WizardLMTeam/WizardMath-7B-V1.1",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    jp_model = AutoModelForCausalLM.from_pretrained(
        "augmxnt/shisa-gamma-7b-v1",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    jp_tokenizer = AutoTokenizer.from_pretrained(
        'augmxnt/shisa-gamma-7b-v1'
    )

    if jp_tokenizer.vocab_size == eng_tokenizer.vocab_size:
        # 除外対象
        skip_layers = []
    else:
        exit(1)
    for k, v in jp_model.state_dict().items():
        if (k in skip_layers) or ("layernorm" in k):
            continue
        chat_vector = math_model.state_dict()[k] - eng_model.state_dict()[k]
        new_v = v + chat_vector.to(v.device)
        v.copy_(new_v)
    new_model_name = 'chat_vector_mistral_7B_math'
    jp_model.save_pretrained(new_model_name)
    jp_tokenizer.save_pretrained(new_model_name)
# MGSM：0.352