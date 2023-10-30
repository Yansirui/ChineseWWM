from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import wandb
from tqdm.auto import tqdm
from transformers import get_scheduler

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name=r'WWMBERT-b8_l3e-05'

wandb.init(
    # set the wandb project where this run will be logged
    project='STS',
    name=model_name,
    # track hyperparameters and run metadata
    config={
        "learning_rate":3e-5 ,
        "architecture": "Transformer",
        "dataset": "Finance",
        "epochs": 3,
        }
)

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained(r'/home/sirui/WMM/Medicine/model/BERT_Model/BERT_BASE')
model = BertForSequenceClassification.from_pretrained(r'/home/sirui/WMM/Medicine/model/Whole_mask/'+model_name,num_labels=2).to(device)

# 加载训练数据和测试数据
with open('/home/sirui/WMM/Medicine/Data/sts_data/CHIP-STS_train.json', 'r', encoding='utf-8') as train_file:
    train_data = json.load(train_file)

with open('/home/sirui/WMM/Medicine/Data/sts_data/CHIP-STS_dev.json', 'r', encoding='utf-8') as dev_file:
    dev_data = json.load(dev_file)

# 准备训练数据
# 请根据数据文件中的格式准备训练数据
# 在这里，我们使用示例数据作为演示

# 准备测试数据
# 请根据数据文件中的格式准备测试数据
# 在这里，我们使用示例数据作为演示

# 定义优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01, eps=1e-8)
criterion = torch.nn.CrossEntropyLoss()


# 初始化最佳指标和模型
best_accuracy = 0.0
best_model = None

# 转换数据为特征
def prepare_data(data):
    input_ids = []
    attention_masks = []
    labels = []
    token_type_ids=[]
    for example in data:
        text1 = example['text1']
        text2 = example['text2']
        label = int(example['label'])

        inputs = tokenizer(text1, text2, return_tensors='pt', padding='max_length', truncation=True,max_length=128)
        input_ids.append(inputs['input_ids'][0])
        attention_masks.append(inputs['attention_mask'][0])
        token_type_ids.append(inputs['token_type_ids'][0])
        labels.append(label)

    input_ids = torch.stack(input_ids).to(device)
    attention_masks = torch.stack(attention_masks).to(device)
    labels = torch.tensor(labels).to(device)
    token_type_ids=torch.stack(token_type_ids).to(device)
    return input_ids, token_type_ids,attention_masks, labels

train_input_ids, train_token_type_ids,train_attention_masks, train_labels = prepare_data(train_data)
test_input_ids, test_token_type_ids,test_attention_masks, test_labels = prepare_data(dev_data)

train_dataset = TensorDataset(train_input_ids, train_token_type_ids,train_attention_masks, train_labels)
test_dataset = TensorDataset(test_input_ids, test_token_type_ids,test_attention_masks, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32)
num_epochs=3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=int(num_training_steps * 0.1),
    num_training_steps=num_training_steps,
)
# 训练和测试循环
model.train()
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        input_ids, token_type_ids,attention_mask, labels = batch
        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        wandb.log({"loss":loss})
        loss.backward()
        total_loss = total_loss + loss
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        progress_bar.set_description_str(f'Epoch{epoch}')
        progress_bar.set_postfix(loss=loss.item())

        if (step + 1) % 250 == 0:
            # 测试当前模型
            model.eval()
            test_predictions = []
            all_labels=[]
            with torch.no_grad():
                for test_batch in test_dataloader:
                    test_input_ids, test_token_type_ids,test_attention_mask, test_labels = test_batch
                    for l in test_labels.cpu().numpy():
                        all_labels.append(l)
                    test_outputs = model(input_ids=test_input_ids, token_type_ids=test_token_type_ids,attention_mask=test_attention_mask)
                    test_logits = test_outputs.logits
                    test_predictions.extend(test_logits.argmax(dim=1).cpu().numpy())
            #print(test_labels)
            #print(test_predictions)
            accuracy = accuracy_score(all_labels, test_predictions)
            precision = precision_score(all_labels, test_predictions)
            recall = recall_score(all_labels, test_predictions)
            f1 = f1_score(all_labels, test_predictions)
            wandb.log({'Accuracy':accuracy})
            wandb.log({'Precision':precision})
            wandb.log({'Recall':recall})
            wandb.log({'F1_Score':f1})
            print(f"Step {step + 1}, Epoch {epoch + 1}, Train Loss: {total_loss}, "
                  f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
            model.train()
model.save_pretrained(r'/home/sirui/WMM/Medicine/model/sts_model/'+model_name)
    #tokenizer.save_pretrained('best_bert_sts_model')
