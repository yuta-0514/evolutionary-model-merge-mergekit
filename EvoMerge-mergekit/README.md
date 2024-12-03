# Evolutionary Model Merge
[Evolutionary Optimization of Model Merging Recipes](https://sakana.ai/evolutionary-model-merge-jp/)

## Install Libraries
```
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit
pip install -e .[evolve]
```

### Prepare Dataset
```Python3
import datasets
ds = datasets.load_dataset("juletxara/mgsm", "ja")
ds["train"].save_to_disk("./mgsm-jp/train")
```

## Model Merge
```
mergekit-evolve ./evol_merge_config.yml \
		--storage-path evol_merge_storage \
		--task-search-path eval_tasks \
		--in-memory \
		--no-merge-cuda --wandb
```

## Save Merged Model
```
mergekit-yaml evol_merge_storage/best_config.yaml merge
```

## Evaluate
```
Python eval_mgsm.py -model_id <YOUR MODEL>
```

| ID | Model | Type | Size | [MGSM-JA](https://huggingface.co/datasets/juletxara/mgsm)(acc↑) |
| -- | -- | -- | -- | -- |
| 1 | [Shisa Gamma 7B v1](https://huggingface.co/augmxnt/shisa-gamma-7b-v1) | JA general | 7B | 6.4 |
| 2 | [WizardMath 7B v1.1](https://huggingface.co/WizardLMTeam/WizardMath-7B-V1.1) | EN math | 7B | 35.2 |
| 3 | [Abel 7B 002](https://huggingface.co/GAIR/Abel-7B-002) | EN math | 7B | 21.2 |
| 4 | [EvoLLM-JP](https://huggingface.co/SakanaAI/EvoLLM-JP-v1-7B) |  | 7B | 41.2 |
| 5 | Merged Model | 1+2+3 | 7B | 30.8 |

### Using part of test data for model merge

| Only Train（8 data） | Train + Test（8+100 data） |
| -- | -- |
| 34.7 | 22.7 |

　* Accuracy of remaining test data

## Citation
- https://sakana.ai/evolutionary-model-merge-jp/
- https://acro-engineer.hatenablog.com/entry/2024/05/07/124507
- https://github.com/openai/simple-evals/tree/main