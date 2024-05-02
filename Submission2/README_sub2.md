# Submission 2

## Info
The model card: [Llama3-70b-fine-tune](https://huggingface.co/naturaltonguetwisters/llama3-70b-alpaca-cleaned)

## Install - HPC

1. Clone the model from HuggingFace into a directory `llama_alpaca_clean`.

2. Build the singularity container:
```
singularity build nlp_pt.sif Singularity
```

## Usage - HPC

Training the model - job:
```
sbatch llama_train.sh
```

Testing the model - job:
```
sbatch llama_test.sh
```
