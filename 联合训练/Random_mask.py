from transformers import BertTokenizer, BertForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments,DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
from datasets import DatasetDict

check_point=r'/home/sirui/WMM/Finance/model/BERT_Model/BERT-wmm-ext'
tokenizer = BertTokenizer.from_pretrained(check_point)
train_file=r"/home/sirui/WMM/Finance/Data/Init_Corpus.txt"
#file=r"/home/sirui/WMM/data/Original_corpus.txt"
#mask_file=r'/home/sirui/WMM/data/mask.txt'
# 准备数据集

#with open(mask_file,'r',encoding='utf-8') as f:
#    mask_word=f.readlines()

#因为没有准备测试数据，所以用的同一个，当然可以将test去掉，只用train也是可以的
Masked_dataset =DatasetDict.from_text({'train':train_file,'test':train_file})
#Original_dataset = DatasetDict.from_text({'train':file,'test':file})

def tokenize_function(example):
    return tokenizer(example["text"],truncation=True,padding=True,return_tensors='pt')

Mask_tokenized=Masked_dataset.map(tokenize_function,batched=True)
#Original_tokenized=Original_dataset.map(tokenize_function,batched=True)
Mask_tokenized=Mask_tokenized.remove_columns(["text"])
Mask_tokenized.set_format("torch")
#Original_tokenized=Original_tokenized.remove_columns(["text"])
#Original_tokenized.set_format("torch")


masked_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
#original_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
masked_dataloader = DataLoader(Mask_tokenized['train'] , shuffle=False , batch_size=8, collate_fn=masked_collator)
#original_dataloader = DataLoader(Original_tokenized['train'] , shuffle=False , batch_size=8, collate_fn=original_collator)

# 定义模型
model = BertForMaskedLM.from_pretrained(check_point)
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)
from transformers import get_scheduler
num_epochs = 3
num_training_steps = num_epochs * len(masked_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))
#progress_bar1 = tqdm(range(num_training_steps),original_dataloader)

model.train()

for epoch in range(num_epochs):
    #batch_num=0
    for batch in masked_dataloader:
        '''
        for i in range(len(batch['labels'])):
            index=batch_num*8+i
            current_index=0
            for j in range(len(batch['labels'][i])):
                if batch['labels'][i][j] != -100:
                    batch['labels'][i][j] = tokenizer.convert_tokens_to_ids(mask_word[index][current_index])
                    current_index=current_index+1
        batch_num=batch_num+1
'''
        batch = {k: v.to(device) for k, v in batch.items()}
        #print(batch)
        outputs = model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],token_type_ids=batch['token_type_ids'],labels=batch['labels'])
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        progress_bar.set_description_str(f'Epoch{epoch}')
        progress_bar.set_postfix(loss=loss.item())

    model.save_pretrained('/home/sirui/WMM/Finance/model/Encoder/RM-BERT'+str(epoch))
