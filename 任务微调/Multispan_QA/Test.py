import pickle
import re
import torch
from torch.utils import data
from tqdm.auto import tqdm
from transformers import BertTokenizerFast
import utils
from model import BertForMultiSpanQuestionAnswering

device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = "/home/sirui/WMM/Knowledge_based_WMM/Extractive_QA/saved_models/macBERT"
model = BertForMultiSpanQuestionAnswering.from_pretrained(
    model_dir, num_labels=3, ignore_mismatched_sizes=True
).to(device)

tokenizer = BertTokenizerFast.from_pretrained(r'/home/sirui/WMM/Finance/model/BERT_Model/BERT_BASE')
special_tokens_dict = {"additional_special_tokens": ["\001"]}
tokenizer.add_special_tokens(special_tokens_dict)
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
# 2. 扩充模型lookup嵌入层词表大小 resize_token_embeddings
model.resize_token_embeddings(len(tokenizer))
inputs = pickle.load(open("/home/sirui/WMM/Law/Data/myData.pkl", "rb"))
querys = inputs["querys"]
contexts = inputs["contexts"]
positive_answer_starts = inputs["pos_answer_starts"]
positive_answer_ends = inputs["pos_answer_ends"]
answer_span_texts = inputs["answer_span_texts"]
train_split = 34299
dev_split = 38299
test_contexts = contexts[dev_split:]
test_querys = querys[dev_split:]
test_positive_answer_starts = positive_answer_starts[dev_split:]
test_positive_answer_ends = positive_answer_ends[dev_split:]
test_positive_answer_span_texts = answer_span_texts[dev_split:]
test_querys_tokenized = tokenizer(test_querys, add_special_tokens=False)
test_contexts_tokenized = tokenizer(test_contexts, add_special_tokens=False)

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
            max_len=450,
            spans_len=30
    ):
        self.split = split
        self.positive_answer_start_seqs = positive_answer_start_seqs
        self.positive_answer_end_seqs = positive_answer_end_seqs
        # self.negative_answer_start1_seqs = negative_answer_start1_seqs
        # self.negative_answer_end1_seqs = negative_answer_end1_seqs
        # self.negative_answer_start2_seqs = negative_answer_start2_seqs
        # self.negative_answer_end2_seqs = negative_answer_end2_seqs

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

        if self.split == "train" or self.split == "test":
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
                    #print("query:", train_querys[idx], train_positive_answer_starts[idx],
                          #train_positive_answer_ends[idx], answer_span_texts[idx])
                    #print("pos_answer_start_token:", pos_answer_start_token)
                    #print("pos_answer_end_token:", pos_answer_end_token)
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

test_set = MultiSpanDataset(
    "test", test_positive_answer_starts, test_positive_answer_ends,
    test_querys_tokenized, test_contexts_tokenized
)
test_batch_size = 4
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
    #print("batched_contexts size:", len(batched_contexts_tokenized))
    for b in range(batch_size):
        offset = tensors_data[1][b].tolist().index(1)
        # logits: (batch_size, num_label, seq_len)
        # _prob, _idx: (batch_size, )
        _prob, _idx = torch.max(outs.logits[b], dim=1)  # (seq_len, num_label)
        # [offset: -1]

        start_end_pairs = extract_start_end_pairs_from_bio_idx(_idx[offset:])
        #print("!!!!start_end_pairs:", start_end_pairs)

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
model.eval()
dev_loss = 0
dev_batch_num = 0
dev_example_num = 0
TP, FP, FN = 0, 0, 0
macro_avg_f1 = 0
macro_avg_precision, macro_avg_recall = 0, 0
with torch.no_grad():
    for i, dev_data in enumerate(tqdm(test_loader)):
        dev_batch_num += 1
        outputs = model(
            input_ids=dev_data[0].to(device),
            token_type_ids=dev_data[1].to(device),
            attention_mask=dev_data[2].to(device),
            positive_start_positions=dev_data[3].to(device),
            positive_end_positions=dev_data[4].to(device),
            seo_targets=dev_data[5].to(device)
        )
        dev_loss += outputs.loss
        predicted_answers, improved_predicted_answers = evaluate(
            dev_data, outputs,
            test_contexts[i * test_batch_size: i * test_batch_size + dev_data[0].shape[0]],
            test_contexts_tokenized[i * test_batch_size: i * test_batch_size + dev_data[0].shape[0]]
        )
        for j, ans in enumerate(predicted_answers):
            dev_example_num += 1
            index = i * test_batch_size + j
            context = test_contexts[index].replace(r"\s+", "")
            gold_answer_set = set(test_positive_answer_span_texts[index]) \
                if test_positive_answer_span_texts[index] is not None else set()
            FN += len(gold_answer_set)
            gold_answer_text = " @ ".join(test_positive_answer_span_texts[index]) \
                if test_positive_answer_span_texts[index] is not None else ""
            gold_answer_span_set = set("".join(test_positive_answer_span_texts[index])) \
                if test_positive_answer_span_texts[index] is not None else set({})
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
print(f"| dev_loss: {dev_loss.item() / dev_batch_num: .3f} "
f"| dev_precision: {precision * 100:.2f} "
f"| dev_recall: {recall * 100:.2f} "
f"| exact_f1 = {f1 * 100:.2f}"
f"| exact_precision = {precision * 100:.2f}"
f"| exact_recall = {recall * 100:.2f}"
f"| macro_avg_f1 = {macro_avg_f1 * 100:.2f}"
f"| macro_avg_precision = {macro_avg_precision * 100:.2f}"
f"| macro_avg_recall = {macro_avg_recall * 100:.2f}")
