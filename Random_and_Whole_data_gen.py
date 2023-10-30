import random
import bisect
import json


def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data

def find_all(string, sub):
    start = 0
    pos = []
    while True:
        start = string.find(sub, start)
        if start == -1:
            return pos
        pos.append(start)
        start += len(sub)

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        result = f.readlines()
    return [line.strip() for line in result]

def whole_mask(sentence, terms):
    terms=sorted(terms,key=lambda i:len(i),reverse=True)
    masked = '♥'
    masked1 = '☕'
    sentence = sentence.replace('\n', '')
    start = []
    end = []
    sentence_masked_terms = []
    sentence_masked_words=[]
    i=0
    for term in terms:
        if term == '':
            break
        length = len(term)
        positions = find_all(sentence, term)
        for pos in positions:
            kequ=True
            for s,e in zip(start,end):
                if(s <  pos and pos <  e) or (s < pos + length and pos + length < e) or (pos < s and pos+length > e) or (pos==s):
                    kequ=False
            if kequ:
                start.append(pos)
                end.append(pos + length)
            else:
                continue
    mask = len(sentence) * 0.15
    start_index = []
    term_length = []
    if len(start) == 0:
        random_int=[]
        while mask > 0 :
            word_num=random.randint(0, len(sentence) - 1)
            while word_num in random_int:
                word_num=random.randint(0, len(sentence) - 1)
            random_int.append(word_num)
            mask=mask-1
        random_int=sorted(random_int)
        sentence_list=list(sentence)
        for index in random_int:
            sentence_list[index]=masked1
            sentence_masked_words.append(sentence[index])
        return ''.join(sentence_list),term_length,sentence_masked_terms,sentence_masked_words


    while mask > 0:
        index = random.randint(0, len(start) - 1)
        word = sentence[start[index]:end[index]]
        mylist = list(sentence)
        j = start[index]
        while j < end[index]:
            mylist[j] = masked
            j += 1
        sentence = ''.join(mylist)
        bisect.insort(start_index, start[index])
        insort_index = start_index.index(start[index])
        term_length.insert(insort_index, end[index] - start[index])
        sentence_masked_terms.insert(insort_index, word)
        mask -= (end[index] - start[index])
        start.pop(index)
        end.pop(index)
        if len(start) == 0:
            break
    random_int=[]
    while mask > 0:
        word_num = random.randint(0, len(sentence) - 1)
        while word_num in random_int or sentence[word_num] == '♥':
            word_num = random.randint(0, len(sentence) - 1)
        random_int.append(word_num)
        mask=mask-1
    if random_int!=[]:
        random_int=sorted(random_int)
        sentence_list = list(sentence)
        for index in random_int:
            sentence_list[index] = masked1
            sentence_masked_words.append(sentence[index])
        return ''.join(sentence_list), term_length, sentence_masked_terms,sentence_masked_words
    else:
        return sentence, term_length, sentence_masked_terms,sentence_masked_words

import re
def split_sentences(sentence):
    MAX_LENGTH = 500
    sentences = []
    sub_sentences=re.split(r'([。！!？?])',sentence)
    #sub_sentences.append("")
    #sub_sentences = ["".join(i) for i in zip(sentences[0::2],sentences[1::2])]
    #sub_sentences = sentence.split('，')
    cur_sentence = ''
    for sub in sub_sentences:
        if len(cur_sentence) + len(sub) > MAX_LENGTH:
            if cur_sentence != '':
                sentences.append(cur_sentence)
                cur_sentence = sub
            else:
                cur_sentence = sub
        else:
            cur_sentence += sub
    if cur_sentence != '':
        sentences.append(cur_sentence)
    return sentences


def add_blank(content):
    sentence=''
    i=0
    length=len(content)
    while i < length:
        if content[i]!= '♥' and content[i] != '☕':
            sentence=sentence+content[i]
            i=i+1
        else:
            sen=''
            while content[i]=='♥' or content[i]=='☕':
                sen=sen+content[i]
                i=i+1
                if i == length:
                    break
            sen=' '+sen+' '
            sentence=sentence+sen
    return sentence
nums=1
term_path=r'/home/sirui/WMM/Medicine/baike_data/WWM_terms.txt'
with open(term_path,'r',encoding='utf-8') as f:
    terms=f.readlines()
for i in range(len(terms)):
    terms[i]=terms[i].replace('\n','')
terms=list(set(terms))
#terms=[]
print(len(terms))
import re
with open(r'/home/sirui/WMM/Medicine/baike_data/baike_corpus.txt', 'r', encoding='utf-8') as f:
    save = []
    all_term_length = []
    all_terms = []
    all_words=[]
    with    open(r'/home/sirui/WMM/Medicine/baike_data/WWM/Masked_corpus.txt', 'w', encoding='utf-8') as f_save, \
            open(r'/home/sirui/WMM/Medicine/baike_data/WWM/masked_terms.txt', 'w', encoding='utf-8') as f_terms, \
            open(r'/home/sirui/WMM/Medicine/baike_data/WWM/terms_length.txt', 'w', encoding='utf-8') as f_length, \
            open(r'/home/sirui/WMM/Medicine/baike_data/WWM/masked_words.txt', 'w', encoding='utf-8') as f_words:
        sentences = (line.strip() for line in f)
        for sentence in sentences:
            sentence = sentence.replace('♥', '').replace('☕','')
            sentence = re.sub(r'\s','',sentence)
            if len(sentence) > 500:
                sub_sentences = split_sentences(sentence)
                for sub_sentence in sub_sentences:
                    if 500 > len(sub_sentence) > 8:
                        masked_sentence, cur_term_length, cur_terms , cur_words= whole_mask(sub_sentence, terms)
                        masked_sentence = add_blank(masked_sentence)
                        save.append(masked_sentence)
                        if cur_term_length==[]:
                            length='None'
                            terms1='None'
                        else:
                            length = '\t'.join(str(length) for length in cur_term_length)
                            terms1 = '\t'.join(cur_terms)
                        if cur_words == []:
                            words='None'
                        else:
                            words='\t'.join(cur_words)
                        all_terms.append(terms1)
                        all_term_length.append(length)
                        all_words.append(words)
            elif 8 < len(sentence) <= 500:
                masked_sentence, cur_term_length, cur_terms, cur_words = whole_mask(sentence, terms)
                masked_sentence = add_blank(masked_sentence)
                save.append(masked_sentence)
                if cur_term_length == []:
                    length = 'None'
                    terms1 = 'None'
                else:
                    length = '\t'.join(str(length) for length in cur_term_length)
                    terms1 = '\t'.join(cur_terms)
                if cur_words == []:
                    words = 'None'
                else:
                    words = '\t'.join(cur_words)
                all_terms.append(terms1)
                all_term_length.append(length)
                all_words.append(words)
            if len(save) % 1000 == 0:
                print('保存数目:',nums*1000)
                nums = nums + 1
                for senn, termm, lengthh ,wordd in zip(save, all_terms, all_term_length,all_words):
                    f_save.write(senn + '\n')
                    f_terms.write(termm + '\n')
                    f_length.write(lengthh + '\n')
                    f_words.write(wordd+'\n')
                save = []
                all_term_length = []
                all_terms = []
                all_words = []
        if len(save) > 0:
            for senn, termm, lengthh ,wordd in zip(save, all_terms, all_term_length,all_words):
                f_save.write(senn + '\n')
                f_terms.write(termm + '\n')
                f_length.write(lengthh + '\n')
                f_words.write(wordd+'\n')




