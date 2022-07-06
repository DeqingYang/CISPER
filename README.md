# CISPER
Codes for **Contextual Information and Commonsense Based Prompt for Emotion Recognition in Conversation**! :blush: :blush: :blush:

## Requiements
`datasets==1.13.3`  
`huggingface-hub==0.0.19`  
`pytorch-lightning==0.8.1`  
`scikit-learn==1.0`  
`scipy==1.7.1`  
`sklearn==0.0`  
`tensorboard==2.7.0`  
`tensorboard-data-server==0.6.1`  
`tensorboard-plugin-wit==1.8.0`  
`termcolor==1.1.0`  
`threadpoolctl==3.0.0`  
`tokenizers==0.10.3`  
`torch==1.9.1`  
`torchaudio==0.9.0a0+a85b239`  
`torchvision==0.10.1`  
`tqdm==4.62.3`  
`transformers==4.11.3`  

## Experimments on MELD
Run the following commands on terminal to run an experiments on dataset MELD
`python prt_mainCOM.py`

## Experimments on EmoryNLP
Run the following commands on terminal to run an experiments on dataset EmoryNLP
`python prt_mainCOM_erm.py`

***You can hange the hyperparameters at the corresponding position in .py files as you like!***
### Tips:
The dataset files: utterance features and commonsense features can be obtain via this url, which is published by the author of COSMIC: https://github.com/declare-lab/conv-emotion. 
The pre-trained language models for roberta-large can be obtain by: `RobertaForMaskedLM.from_pretrained("roberta-large")` instead of local files.
