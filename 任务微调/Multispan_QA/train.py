"""
BIO tagging
O=1, B=2, I=4
(O, O) -> 3
(O, B) -> 4
(B, I) (I, I) -> 8 12
(B, B) (I, B) -> 6 10
(B, O) (I, O) -> 5 9
在v14的基础上 添加overlap F1 score metric

========== RUN1 =========
Run summary:
wandb:        dev_f1 0.38832
wandb:      dev_loss 0.17645
wandb: dev_precision 0.31764
wandb:    dev_recall 0.49945
wandb:            lr 0.0
wandb:          step 2550
wandb:     train_acc 0.0
wandb:    train_loss 0.02376

"""
import datetime
import os
import pickle
import re
import glob

import torch
#import wandb
from torch.optim import AdamW
from torch.utils import data
from tqdm.auto import tqdm
from transformers import BertTokenizerFast
from model import BertForMultiSpanQuestionAnswering
from transformers import get_scheduler
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup

import utils

#del os.environ["http_proxy"]
#del os.environ["https_proxy"]

#os.environ["WANDB_BASE_URL"] = "http://10.11.195.12:8888"
#os.environ["WANDB_API_KEY"] = "local-93c0612d9c281363acccdfb79eb419697003d0e5"
'''
wandb.init(
    project="multi-span-qa",
    name="freeze-bert-layers",
    entity="luozhiyi",
    save_code=True
)'''
root_name = os.path.basename(os.path.dirname(__file__))
#code_artifact = wandb.Artifact("-".join([root_name, '230115']), type='code')
#for path in glob.glob("**/*.py", recursive=True):
#    if path.startswith("wandb/"):
#        continue
    #code_artifact.add_file(path)
#wandb.run.log_artifact(code_artifact)

# Capture a dictionary of hyper parameters with config
config = {
    "learning_rate": 1e-4,
    "num_epoch": 10,
    # training batch_size
    "batch_size": 8
}
# Validation | Epoch 5 | dev_loss:  0.150 |
# exact_f1 = 31.49| exact_precision = 24.92| exact_recall = 42.76|
# macro_avg_f1 = 59.68| macro_avg_precision = 64.37| macro_avg_recall = 65.98

# Validation | Epoch 10 | dev_loss:  0.219 |
# exact_f1 = 36.33| exact_precision = 28.73| exact_recall = 49.39|
# macro_avg_f1 = 65.52| macro_avg_precision = 67.48| macro_avg_recall = 74.02

# Roberta
# Validation | Epoch 10 | dev_loss:  0.218 |
# exact_f1 = 40.54| exact_precision = 33.62| exact_recall = 51.05|
# macro_avg_f1 = 67.70| macro_avg_precision = 70.81| macro_avg_recall = 75.68
device = "cuda" if torch.cuda.is_available() else "cpu"

# Change "fp16_training" to True to support automatic mixed precision training (fp16)
fp16_training = False

if fp16_training:
    from accelerate import Accelerator

    accelerator = Accelerator(fp16=True)
    device = accelerator.device
# Documentation for the toolkit:  https://huggingface.co/docs/accelerate/

# 构造模型
# model_dir = "/home/luozhiyi/projects/nlpclouds/huggingface/bert-base-chinese/"
model_dir = "/home/sirui/WMM/Law/model/BERT_Model/BERT_BASE"
# model = BertForMultiSpanQuestionAnswering.from_pretrained(
#     model_dir, num_labels=2
# ).to(device)
model = BertForMultiSpanQuestionAnswering.from_pretrained(
    model_dir, num_labels=3, ignore_mismatched_sizes=True
).to(device)

# == 构造分词器 ==
tokenizer = BertTokenizerFast.from_pretrained(model_dir)
# 1. 为分词器加入特殊词
special_tokens_dict = {"additional_special_tokens": ["\001"]}
tokenizer.add_special_tokens(special_tokens_dict)
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
# 2. 扩充模型lookup嵌入层词表大小 resize_token_embeddings
model.resize_token_embeddings(len(tokenizer))

# == 数据处理 ==
inputs = pickle.load(open("/home/sirui/WMM/Law/Data/EXQA_Data.pkl", "rb"))
querys = inputs["querys"]
contexts = inputs["contexts"]
positive_answer_starts = inputs["pos_answer_starts"]
positive_answer_ends = inputs["pos_answer_ends"]
answer_span_texts = inputs["answer_span_texts"]

# 数据划分：train/dev/test split
train_split = 32991
dev_split = 36968

train_querys = querys[:train_split]
dev_querys = querys[train_split: dev_split]
test_querys = querys[dev_split:]

train_contexts = contexts[:train_split]
dev_contexts = contexts[train_split: dev_split]
test_contexts = contexts[dev_split:]

train_positive_answer_starts = positive_answer_starts[:train_split]
train_positive_answer_ends = positive_answer_ends[:train_split]
# train_negative_answer_starts_1 = negative_answer_starts_1[:train_split]
# train_negative_answer_ends_1 = negative_answer_ends_1[:train_split]
# train_negative_answer_starts_2 = negative_answer_starts_2[:train_split]
# train_negative_answer_ends_2 = negative_answer_ends_2[:train_split]

dev_positive_answer_starts = positive_answer_starts[train_split: dev_split]
dev_positive_answer_ends = positive_answer_ends[train_split: dev_split]
#dev_negative_answer_starts_1 = negative_answer_starts_1[train_split: dev_split]
#dev_negative_answer_ends_1 = negative_answer_ends_1[train_split: dev_split]
#dev_negative_answer_starts_2 = negative_answer_starts_2[train_split: dev_split]
#dev_negative_answer_ends_2 = negative_answer_ends_2[train_split: dev_split]

test_positive_answer_starts = positive_answer_starts[dev_split:]
test_positive_answer_ends = positive_answer_ends[dev_split:]
#test_negative_answer_starts_1 = negative_answer_starts_1[dev_split:]
#test_negative_answer_ends_1 = negative_answer_ends_1[dev_split:]
#test_negative_answer_starts_2 = negative_answer_starts_2[dev_split:]
#test_negative_answer_ends_2 = negative_answer_ends_2[dev_split:]
#
dev_positive_answer_span_texts = answer_span_texts[train_split: dev_split]
test_positive_answer_span_texts = answer_span_texts[dev_split:]

# 3. 分词
train_querys_tokenized = tokenizer(train_querys, add_special_tokens=False)
dev_querys_tokenized = tokenizer(dev_querys, add_special_tokens=False)
test_querys_tokenized = tokenizer(test_querys, add_special_tokens=False)

train_contexts_tokenized = tokenizer(train_contexts, add_special_tokens=False)
dev_contexts_tokenized = tokenizer(dev_contexts, add_special_tokens=False)
test_contexts_tokenized = tokenizer(test_contexts, add_special_tokens=False)


# 定义数据集
def char_to_token(char_idx, tokenized_text, offset):
    if char_idx >= 0:
        token_idx = tokenized_text.char_to_token(char_idx)
    else:
        token_idx = char_idx
    # """
    if token_idx is None:
        return char_to_token(char_idx-1, tokenized_text, offset)
    elif token_idx < 0:
        token_idx = 0
    #"""
    #if token_idx < 0:
    #    token_idx = 0
    #"""
    else:
        token_idx += offset
    return token_idx


class MultiSpanDataset(data.Dataset):
    def __init__(
            self, split,
            positive_answer_start_seqs, positive_answer_end_seqs,
            tokenized_querys,
            tokenized_contexts,
            max_len=350,
            spans_len=20
    ):
        self.split = split
        self.positive_answer_start_seqs = positive_answer_start_seqs
        self.positive_answer_end_seqs = positive_answer_end_seqs
        #self.negative_answer_start1_seqs = negative_answer_start1_seqs
        #self.negative_answer_end1_seqs = negative_answer_end1_seqs
        #self.negative_answer_start2_seqs = negative_answer_start2_seqs
        #self.negative_answer_end2_seqs = negative_answer_end2_seqs

        self.tokenized_querys = tokenized_querys
        self.tokenized_contexts = tokenized_contexts
        self.SENT_SPLIT_REGEX = re.compile('([﹒﹔﹖﹗．；。！？\n]["’”」』]{0,2}|：(?=["‘“「『]{1,2}|$))')
        self.max_len = max_len
        self.spans_len = spans_len

    def __len__(self):
        return len(self.positive_answer_start_seqs)

    def __getitem__(self, idx):
        positive_answer_start_seq = self.positive_answer_start_seqs[idx]
        positive_answer_end_seq = utils.subtract_offset_for_nested_list_elements(
            self.positive_answer_end_seqs[idx], 1
        )

        tokenized_query = self.tokenized_querys[idx]
        tokenized_context = self.tokenized_contexts[idx]

        input_ids_query = [101] + tokenized_query.ids + [102]
        input_ids_context = tokenized_context.ids + [102]

        # Pad sequence and obtain inputs to model
        input_ids, token_type_ids, attention_mask = \
            self.padding(input_ids_query, input_ids_context, return_valid_len=False)

        real_seo_target = torch.zeros_like(torch.tensor(input_ids))
        ignored_index = torch.tensor(input_ids).size(0) - 1

        if self.split == "train" or self.split == "dev":
            positive_answer_start_token_seq, positive_answer_end_token_seq = [], []

            for k, pos_answer_start in enumerate(positive_answer_start_seq):
                pos_answer_end = positive_answer_end_seq[k]

                try:
                    pos_answer_start_token = char_to_token(pos_answer_start, tokenized_context, len(input_ids_query))
                    pos_answer_end_token = char_to_token(pos_answer_end, tokenized_context, len(input_ids_query))

                    positive_answer_start_token_seq.append(pos_answer_start_token)
                    positive_answer_end_token_seq.append(pos_answer_end_token)

                    real_seo_target.scatter_(
                        0, torch.arange(pos_answer_start_token, pos_answer_end_token + 1).clamp(0, ignored_index), 2
                    ).scatter_(
                        0, torch.arange(pos_answer_start_token, pos_answer_start_token + 1).clamp(0, ignored_index), 1
                    )
                except:
                    print("idx:", idx)
                    print("query:", train_querys[idx], train_positive_answer_starts[idx], train_positive_answer_ends[idx], answer_span_texts[idx])
                    print("pos_answer_start_token:", pos_answer_start_token)
                    print("pos_answer_end_token:", pos_answer_end_token)
                    exit(0)
                

            positive_answer_start_tokens = self.padding_token_seq(positive_answer_start_token_seq)
            positive_answer_end_tokens = self.padding_token_seq(positive_answer_end_token_seq)

            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), \
                   torch.tensor(positive_answer_start_tokens), torch.tensor(positive_answer_end_tokens), \
                   real_seo_target
                   
        # Testing
        else:
            return torch.tensor(input_ids), torch.tensor(token_type_ids), \
                   torch.tensor(attention_mask)

    def padding(self, input_ids_query, input_ids_context, return_valid_len=False):
        # TODO: 实现 truncation
        # Pad zeros if sequence length is shorter than max_len
        padding_len = self.max_len - len(input_ids_query) - len(input_ids_context)
        if padding_len < 0:
            input_ids = (input_ids_query + input_ids_context)[:self.max_len - 1] + [input_ids_context[-1]]
            token_type_ids = ([0] * len(input_ids_query) + [1] * len(input_ids_context))[:self.max_len]
            attention_mask = ([1] * (len(input_ids_query) + len(input_ids_context)))[:self.max_len]
        else:
            input_ids = input_ids_query + input_ids_context + [0] * padding_len
            token_type_ids = [0] * len(input_ids_query) + [1] * len(input_ids_context) + [0] * padding_len
            attention_mask = [1] * (len(input_ids_query) + len(input_ids_context)) + [0] * padding_len

        if return_valid_len:
            valid_len = min(self.max_len, len(input_ids_query) + len(input_ids_context))
            return input_ids, token_type_ids, attention_mask, valid_len

        return input_ids, token_type_ids, attention_mask

    def padding_token_seq(self, token_seq):
        padding_len = self.spans_len - len(token_seq)
        if padding_len < 0:
            input_token_seq = token_seq[:self.spans_len]
        else:
            input_token_seq = token_seq + [self.max_len] * padding_len
        return input_token_seq


# 构造数据集及加载器
train_set = MultiSpanDataset(
    "train", train_positive_answer_starts, train_positive_answer_ends,
    train_querys_tokenized, train_contexts_tokenized
)
dev_set = MultiSpanDataset(
    "dev", dev_positive_answer_starts, dev_positive_answer_ends,
    dev_querys_tokenized, dev_contexts_tokenized
)
test_set = MultiSpanDataset(
    "test", test_positive_answer_starts, test_positive_answer_ends,
    test_querys_tokenized, test_contexts_tokenized
)

train_batch_size = config.get("batch_size", 4)
dev_batch_size = 4
test_batch_size = 4

train_loader = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
dev_loader = data.DataLoader(dev_set, batch_size=dev_batch_size, shuffle=False, pin_memory=True)
test_loader = data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, pin_memory=True)


def token_encoding_to_surface_text(word_id, token_encoding, orig_text):
    if token_encoding.tokens[word_id] == "[UNK]":
        start, end = token_encoding.offsets[word_id]
        return orig_text[start: end]
    else:
        token_start_id = token_encoding.word_ids.index(word_id)
        token_end_id = len(token_encoding.word_ids) - token_encoding.word_ids[::-1].index(word_id) - 1
        start = token_encoding.offsets[token_start_id][0]
        end = token_encoding.offsets[token_end_id][1]
        return orig_text[start: end]


def word_ids_to_surface_text(word_ids, token_encoding, orig_text):
    res = []
    last_word_id = -1
    for word_id in word_ids:
        if word_id == last_word_id:
            continue
        res.append(token_encoding_to_surface_text(word_id, token_encoding, orig_text))
        last_word_id = word_id
    return res


def extract_start_end_pairs_from_idx(_idx):
    results = []
    answer_positions = (_idx > 0).nonzero().squeeze().tolist()
    if isinstance(answer_positions, int):
        answer_positions = [answer_positions]
    last_hit = None
    for pos in answer_positions:
        if last_hit is None:
            results.append(pos)
        elif pos != last_hit + 1:
            results.extend([last_hit, pos])  # add last_end, new_start
        last_hit = pos
    if len(results):
        results.append(last_hit)
    return [results[_: _ + 2] for _ in range(0, len(results), 2)]


def extract_start_end_pairs_from_bio_idx(_idx):
    # O = 0, B = 1, I = 2
    results = []
    answer_positions = (_idx > 0).nonzero().squeeze().tolist()
    if isinstance(answer_positions, int):
        answer_positions = [answer_positions]
    pos_val_pairs = [(pos, _idx[pos]) for pos in answer_positions]
    last_hit = None
    for pos, val in pos_val_pairs:
        if last_hit is None:
            results.append(pos)
        elif pos != last_hit + 1 or val == 1:
            results.extend([last_hit, pos])  # add last_end, new_start
        last_hit = pos
    if len(results):
        results.append(last_hit)
    return [results[_: _ + 2] for _ in range(0, len(results), 2)]


# Function for Evaluation
def evaluate(tensors_data, outs, batched_contexts, batched_contexts_tokenized):
    # data: List [input_ids Tensor, token_type_ids Tensor, attention_mask Tensor,
    # positive_answer_start_tokens, positive_answer_end_tokens,
    # negative_answer_start1_tokens, positive_answer_end_tokens,
    # negative_answer_start2_tokens, negative_answer_end2_tokens, seo_target, valid_span_num]
    batched_answers = []
    batched_improved_answers = []
    batch_size, max_len = tensors_data[0].shape
    print("batched_contexts size:", len(batched_contexts_tokenized))
    for b in range(batch_size):
        offset = tensors_data[1][b].tolist().index(1)
        # logits: (batch_size, num_label, seq_len)
        # _prob, _idx: (batch_size, )
        _prob, _idx = torch.max(outs.logits[b], dim=1)  # (seq_len, num_label)
        # [offset: -1]

        start_end_pairs = extract_start_end_pairs_from_bio_idx(_idx[offset:])
        print("!!!!start_end_pairs:", start_end_pairs)

        qas = []
        improved_qas = []
        for start_pos, end_pos in start_end_pairs:
            start_idx = start_pos + offset
            end_idx = end_pos + offset
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            if end_idx < start_idx:
                answer = None
            else:
                # [CLS] query [SEP] context [SEP]
                answer = tokenizer.decode(tensors_data[0][b][start_idx: end_idx + 1])

            if answer:
                # Remove spaces in answer (e.g. "大 金" --> "大金")
                qas.append(answer.replace(' ', ''))
                # batched_contexts_tokenized[b].word_ids()
                improved_answer = "".join(word_ids_to_surface_text(
                    batched_contexts_tokenized[b].word_ids[start_pos: end_pos + 1],
                    batched_contexts_tokenized[b],
                    batched_contexts[b]
                ))
                improved_qas.append(improved_answer)

        batched_answers.append(qas)
        batched_improved_answers.append(improved_qas)

    return batched_answers, batched_improved_answers


# == 训练 ==
# Training
num_epoch = config.get("num_epoch", 5)
validation = True
logging_step = 100
# Create an optimizer and learning rate scheduler to fine-tune the model.
# Let’s use the AdamW optimizer from PyTorch
# optimizer
learning_rate = config.get("learning_rate", 1e-4)
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_step = num_epoch * len(train_loader)
# schedule
# lr_init = d**(-0.5) * min(step_num**(-0.5), step_num*(warmup_steps**(-1.5)))
# d = 768, step_num = num_epoch * len(train_loader)
# lr_init = 0.00046898 # i.e. 4.69e-4
lr_scheduler = get_scheduler(
    name='linear', optimizer=optimizer,
    num_warmup_steps=200, num_training_steps=num_training_step
)

cos_restarts_lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=200, num_training_steps=num_training_step, num_cycles=4
)


def get_schedulers(decay, num_warmup_steps=200, lr=learning_rate, lower_bound=1e-6):
    schedulers = []
    lrs, opts = [], []
    cur_lr = lr
    while cur_lr > lower_bound:
        lrs.append(cur_lr)
        cur_lr *= decay
    num_training_steps_per_scheduler = num_training_step // len(lrs)

    for lr in lrs:
        opt = AdamW(model.parameters(), lr=lr)
        opts.append(opt)
        schedulers.append(
            get_scheduler(
                name='linear', optimizer=opt,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps_per_scheduler
            )
        )

    last_opt = AdamW(model.parameters(), lr=lower_bound)
    opts.append(last_opt)
    schedulers.append(
        get_scheduler('constant', optimizer=last_opt)
    )
    return schedulers, opts, num_training_steps_per_scheduler


linear_schedulers, opts, per_schedule_steps = get_schedulers(0.5)

if fp16_training:
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

model.train()

print("Start Training...")

total_step = 0
cur_scheduler = linear_schedulers.pop(0)
cur_optimizer = opts.pop(0)

log_str = ""

for epoch in range(num_epoch):
    step = 1
    train_loss = train_acc = 0
    for data in tqdm(train_loader):
        total_step += 1
        if total_step % per_schedule_steps == 0:
            cur_scheduler = linear_schedulers.pop(0)
            cur_optimizer = opts.pop(0)
        # Load all data into GPU
        data = [i.to(device) for i in data]
        outputs = model(
            input_ids=data[0], token_type_ids=data[1], attention_mask=data[2],
            positive_start_positions=data[3], positive_end_positions=data[4], seo_targets=data[5]
        )
        train_loss += outputs.loss
        if fp16_training:
            accelerator.backward(outputs.loss)
        else:
            outputs.loss.backward()

        cur_optimizer.step()
        cur_scheduler.step()
        cur_optimizer.zero_grad()
#wandb.log({"step": len(train_loader) * epoch + step})
        #wandb.log(
        #    {
        #        "train_loss": train_loss.item() / step,
        #        "train_acc": train_acc / (logging_step if step % logging_step == 0 else step % logging_step),
        #        "lr": cur_optimizer.param_groups[0]['lr']
        #    }
        #)
        # TODO: Apply linear learning rate decay
        # Print training loss and accuracy over past logging step
        if step % logging_step == 0:
            # Log metrics inside your training loop to visualize model performance
            loss = train_loss.item() / logging_step
            acc = train_acc / logging_step
            # 为什么放在这里不行呢？
            # wandb.log({"train_loss": loss, "train_acc": acc})
            # wandb.watch(model)
            print(
                f"Epoch {epoch + 1} | Step {step} | loss = {loss:.3f}, acc = {train_acc / logging_step:.3f}"
            )
            train_loss = train_acc = 0

            if validation:
                print("Evaluating Dev Set ...")
                model.eval()
                dev_loss = 0
                dev_batch_num = 0
                dev_example_num = 0
                TP, FP, FN = 0, 0, 0
                macro_avg_f1 = 0
                macro_avg_precision, macro_avg_recall = 0, 0
                with torch.no_grad():
                    for i, dev_data in enumerate(tqdm(dev_loader)):
                        dev_batch_num += 1
                        outputs = model(
                            input_ids=dev_data[0].to(device),
                            token_type_ids=dev_data[1].to(device),
                            attention_mask=dev_data[2].to(device),
                            positive_start_positions=dev_data[3].to(device),
                            positive_end_positions=dev_data[4].to(device),
                            seo_targets=dev_data[5].to(device)
                        )

                        # outputs.logits: (batch_size, num_label, seq_len)
                        dev_loss += outputs.loss
                        # prediction is correct only if answer text exactly matches
                        predicted_answers, improved_predicted_answers = evaluate(
                            dev_data, outputs,
                            dev_contexts[i * dev_batch_size: i * dev_batch_size + dev_data[0].shape[0]],
                            dev_contexts_tokenized[i * dev_batch_size: i * dev_batch_size + dev_data[0].shape[0]]
                        )
                        print(f"#### batch {i} ####")
                        for j, ans in enumerate(predicted_answers):
                            dev_example_num += 1
                            index = i * dev_batch_size + j
                            context = dev_contexts[index].replace(r"\s+", "")
                            gold_answer_set = set(dev_positive_answer_span_texts[index]) \
                                if dev_positive_answer_span_texts[index] is not None else set()
                            FN += len(gold_answer_set)
                            gold_answer_text = " @ ".join(dev_positive_answer_span_texts[index]) \
                                if dev_positive_answer_span_texts[index] is not None else ""
                            gold_answer_span_set = set("".join(dev_positive_answer_span_texts[index])) \
                                if dev_positive_answer_span_texts[index] is not None else set({})
                            print(f"=== 预测第{j}条数据：", f"问题：{dev_querys[index]}",
                                  f"回答片段：{context}\n", "标准答案：",
                                  gold_answer_text, "预测答案：", " @ ".join(ans),
                                  "改善预测答案：", " @ ".join(improved_predicted_answers[j]))
                            log_str += "[{}]\nQuestion: {}\nContext:{}\n\nGround-truth:{}\nPredicted:{}\n\n" \
                                .format(index, dev_querys[index], context, gold_answer_text,
                                        " @ ".join(improved_predicted_answers[j]))
                            predicted_answer_span_set = set({})
                            for improved_ans in improved_predicted_answers[j]:
                                predicted_answer_span_set.update(set(improved_ans))
                                if improved_ans in gold_answer_set:
                                    TP += 1
                                    FN -= 1
                                else:
                                    FP += 1
                            overlap_r = len(predicted_answer_span_set & gold_answer_span_set) / \
                                        len(gold_answer_span_set)
                            overlap_p = len(predicted_answer_span_set & gold_answer_span_set) / \
                                        len(predicted_answer_span_set) if len(predicted_answer_span_set) else 0.0
                            overlap_f1 = 2 * overlap_p * overlap_r / (overlap_p + overlap_r + 1e-9) \
                                if (overlap_p + overlap_r) > 1e-9 else 0.0
                            # overlap_f1 = 2 * overlap_p * overlap_r / (overlap_p + overlap_r + 1e-9) \
                            #     if (overlap_p + overlap_r) < 1e-9 else 2 * overlap_p * overlap_r / (
                            #             overlap_p + overlap_r)
                            macro_avg_f1 += overlap_f1
                            macro_avg_precision += overlap_p
                            macro_avg_recall += overlap_r

                macro_avg_f1 /= dev_example_num
                macro_avg_precision /= dev_example_num
                macro_avg_recall /= dev_example_num

                precision = TP / (TP + FP) if TP + FP else 0.0
                recall = TP / (TP + FN)
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                # dev_loss 得除以steps数目 猜得到每个step即batch的平均loss
                print(f"Validation | Epoch {epoch + 1} "
                      f"| dev_loss: {dev_loss.item() / dev_batch_num: .3f} "
                      f"| dev_precision: {precision * 100:.2f} "
                      f"| dev_recall: {recall * 100:.2f} "
                      f"| exact_f1 = {f1 * 100:.2f}"
                      f"| exact_precision = {precision * 100:.2f}"
                      f"| exact_recall = {recall * 100:.2f}"
                      f"| macro_avg_f1 = {macro_avg_f1 * 100:.2f}"
                      f"| macro_avg_precision = {macro_avg_precision * 100:.2f}"
                      f"| macro_avg_recall = {macro_avg_recall * 100:.2f}")
                #wandb.log({"dev_loss": dev_loss.item() / dev_batch_num})
                #wandb.log({"dev_precision": precision})
                #wandb.log({"dev_recall": recall})
                #wandb.log({"dev_f1": f1})
                #wandb.log({"dev_macro_avg_f1": macro_avg_f1})
                model.train()
                # for param in model.bert.encoder.layer[:11]:
                #     param.requires_grad = False

        step += 1

# Save a model and its configuration file to the directory 「saved_model」
# i.e. there are two files under the directory 「saved_model」: 「pytorch_model.bin」 and 「config.json」
# Saved model can be re-loaded using 「model = BertForQuestionAnswering.from_pretrained("saved_model")」
print("Saving Model ...")
time_signature = datetime.datetime.now().strftime("%Y%m%d%H%M")

os.makedirs("saved_models", exist_ok=True)
model_save_dir = "saved_models/{}".format(time_signature)
model.save_pretrained(model_save_dir)

os.makedirs("test_results", exist_ok=True)
result_save_path = "test_results/{}".format(time_signature)
log_save_path = "4human_eval_log_{}.txt".format(time_signature)
with open(log_save_path, "w") as outf:
    outf.write(log_str)
