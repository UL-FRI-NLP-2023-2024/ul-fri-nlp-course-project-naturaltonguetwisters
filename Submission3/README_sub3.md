# Submission 3

## Models
Llama 3 8b used: [Llama3-8b-fine-tune](https://huggingface.co/naturaltonguetwisters/s3-llama3-8b-alpaca-clean)
Llama 3 70b used: [Llama3-70b-fine-tune](https://huggingface.co/naturaltonguetwisters/s3-llama3-70b-alpaca-clean)
Mistral 7b v0.2 used: [Mistral-7b-v02-fine-tune](https://huggingface.co/naturaltonguetwisters/s3-mistral-7b-v02-alpaca-clean)

## Install - HPC

1. Clone the models from HuggingFace into directories:
- `llama3_8b_alpaca_clean`
- `llama3_70b_alpaca_clean`
- `mistral_7b_v02_alpaca_clean`

2. Build the singularity container:
```
singularity build nlp_pt.sif Singularity
```

## Training

To train the models, use the following commands:
```
sbatch llama8b_train.sh
sbatch llama70b_train.sh
sbatch mistral7b_v02_train.sh
```

## Usage - HPC

To test the models, run the following commands:
```
sbatch llama8b_inference.sh
sbatch llama70b_inference.sh
sbatch mistral7b_v02_inference.sh
```

Commands above run the models on the tests defined in `tests.json` file, where each entry has to have the properties `character_name`, `novel_title` and `prompt`. When the inference is finished, the results are visible in the `logs` folder under a name that has the model used, inference and the job id (example of a job log: `llama3-8b-inference-<log_id>.out`). 
