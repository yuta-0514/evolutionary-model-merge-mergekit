import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


device = "cuda"
model_id = "evol_merge"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

question = "ジャネットのアヒルは1日に16個の卵を生みます。ジャネットは毎朝朝食の一環で3個を消費し、毎日4個使って友達向けにマフィンを焼きます。残りを市場で1個あたり2ドルの価格で売ります。彼女は毎日市場でいくら手に入れていますか？"
prompt = f"次の数学の問題を解いてください。最終的な答えを出す前に、解答の推論過程を記述してください。答えには整数の答え以外何も追加しないでください。\n問題:{question}\n答え:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
tokens = model.generate(
  **inputs,
  max_new_tokens=2048,
  temperature=0.75,
  top_p=0.95,
  do_sample=True,
)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
