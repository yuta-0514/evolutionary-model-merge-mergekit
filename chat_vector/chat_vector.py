import os
import argparse
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def main(args):
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        token=HF_TOKEN
    )
    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model
    )
    skilled_model = AutoModelForCausalLM.from_pretrained(
        args.skilled_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    cp_model = AutoModelForCausalLM.from_pretrained(
        args.cp_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    cp_tokenizer = AutoTokenizer.from_pretrained(
        args.cp_model
    )

    if cp_tokenizer.vocab_size == base_tokenizer.vocab_size:
        # Excluded　module
        skip_layers = []
    else:
        exit(1)
    for k, v in cp_model.state_dict().items():
        if (k in skip_layers) or ("layernorm" in k):
            continue
        chat_vector = skilled_model.state_dict()[k] - base_model.state_dict()[k]
        new_v = v + chat_vector.to(v.device)
        v.copy_(new_v)
    cp_model.save_pretrained(args.new_model_name)
    cp_tokenizer.save_pretrained(args.new_model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a new model using Chat Vector.')
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model name")
    parser.add_argument("--skilled_model", type=str, default="WizardLMTeam/WizardMath-7B-V1.1", help="Skilled model name")
    parser.add_argument("--cp_model", type=str, default="augmxnt/shisa-gamma-7b-v1", help="CP model name")
    parser.add_argument("--new_model", type=str, default="chat_vector_mistral_7B_math", help="New model name")
    args = parser.parse_args()
    main(args)