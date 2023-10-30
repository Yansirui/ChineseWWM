from model_class import Random_Whole_model
from Processor import *
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import get_scheduler
from tqdm.auto import tqdm
import numpy as np
#tokenizer=AutoTokenizer.from_pretrained(r'D:\Pycharm\基于类的掩码预训练\model\macBERT')
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--pretrained_lr', type=float, default=5e-5, help='Pretrained learning rate')
parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')

# 解析命令行参数
args = parser.parse_args()

# 使用解析后的参数
batch_size = args.batch_size
pretrained_lr = args.pretrained_lr
num_epochs = args.num_epochs

wandb.init(
    # set the wandb project where this run will be logged
    project="WholeMask_Project",
    name = 'RMBERT-B{}_L{}'.format(batch_size,pretrained_lr),
    # track hyperparameters and run metadata
    config={
        "learning_rate":5e-5 ,
        "architecture": "Transformer",
        "dataset": "Finance",
        "epochs": 3,
            }
)


#batch_size = 16
#num_epochs = 3
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model=Random_Whole_model()
sentence_path=r'/home/sirui/WMM/Medicine/baike_data/WWM/Masked_corpus.txt'
terms_path=r'/home/sirui/WMM/Medicine/baike_data/WWM/masked_terms.txt'
length_path=r'/home/sirui/WMM/Medicine/baike_data/WWM/terms_length.txt'
words_path=r'/home/sirui/WMM/Medicine/baike_data/WWM/masked_words.txt'
dataset=Knowledge_Dataset(sentence_path=sentence_path,term_path=terms_path,length_path=length_path,word_path=words_path)
dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
pretrained_params = []
new_params = []
for name, param in model.named_parameters():
    if 'bert' in name or 'mlm' in name:
        pretrained_params.append(param)
    else:
        print(name)
        new_params.append(param)
#pretrained_lr = 5e-5
new_lr=pretrained_lr
import torch.optim as optim
optimizer = optim.Adam([
    {'params': pretrained_params, 'lr': pretrained_lr},
    {'params': new_params, 'lr': new_lr}
])
num_training_steps = num_epochs * len(dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
model.to(device)
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs=model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'],  labels=batch['labels'])
        loss = outputs.loss
        wandb.log({"loss":loss})
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        progress_bar.set_description_str(f'Epoch{epoch}')
        progress_bar.set_postfix(loss=loss.item())
#model.save_pretrained(r'/home/sirui/WMM/Medicine/model/Random_mask/RMBERTwithcls-b{}_l{}'.format(batch_size,pretrained_lr))
model.bert.save_pretrained(r'/home/sirui/WMM/Medicine/model/Whole_mask/RMBERT-b{}_l{}'.format(batch_size,pretrained_lr))
wandb.finish()





