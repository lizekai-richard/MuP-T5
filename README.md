# MuP-T5
Fine-tune T5 on Mup dataset

Please download pre-trained T5 model and LongT5 model from Huggingface.

Dataset can be downloaded from https://github.com/allenai/mup.

To start fine-tuning, run script

```
python3 main.py 
--dataset DATASET_PATH 
--model_config YOUR_PRE_TRAINED_MODEL_PATH 
--init_le YOUR_INIT_LEARNING_RATE 
--batch_size YOUR_BATCH_SIZE 
--lr_decay_rate YOUR_LR_DECAY_RATE 
--num_epochs YOUR_NUMBER_OF_EPOCHS
```

