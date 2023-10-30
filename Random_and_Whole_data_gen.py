import random
import bisect
import json

#读取json文件
def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data

#找到sub在string中的所有起始位置
def find_all(string, sub):
    start = 0
    pos = []
    while True:
        start = string.find(sub, start)
        if start == -1:
            return pos
        pos.append(start)
        start += len(sub)

#读取txt文件，去掉多余符号
def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        result = f.readlines()
    return [line.strip() for line in result]

#根据所给terms列表返回被掩码的sentence(eg:terms=['北京'] --> 我想要去北京去看北京天安门 --> 我想要去♥♥去看♥♥天安门)，同时如果掩码率不足0.15，则采用随机掩码策略
def whole_mask(sentence, terms):
    #term按照长度排序，比如北京天安门，如果词表中有北京天安门，应该优先考虑北京天安门而不是北京
    terms=sorted(terms,key=lambda i:len(i),reverse=True)
    #分别为词掩码符号和token掩码符号
    masked = '♥'
    masked1 = '☕'
    sentence = sentence.replace('\n', '')
    #记录起始位置，用于判断是否找到重合的词汇：如北京天安门，我已经找到北京天安门了，但是我还有'北京'这个词汇，利用start记录判断当前北京词汇是否用于其他词汇
    #为什么不直接找到该词汇就用masked替换呢，因为要控制掩码率！
    start = []
    end = []
    #被掩码的terms列表以及token列表
    sentence_masked_terms = []
    sentence_masked_words=[]
    i=0
    for term in terms:
        if term == '':
            break
        length = len(term)
        #找到词汇起始位置
        positions = find_all(sentence, term)
        for pos in positions:
            #是否可以作为掩码的候选词汇，True代表可以
            kequ=True
            for s,e in zip(start,end):
                #不可以作为候选词汇的条件，与已选词汇发生重叠
                if(s <  pos and pos <  e) or (s < pos + length and pos + length < e) or (pos < s and pos+length > e) or (pos==s):
                    kequ=False
            if kequ:
                start.append(pos)
                end.append(pos + length)
            else:
                continue
    #掩码率：0.15，可自己修改
    mask = len(sentence) * 0.15
    start_index = []
    term_length = []
    
    #如果该句没有词表内词汇，则选择用随机掩码
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

    #有词表中的词汇
    while mask > 0:
        #随机挑选某个词
        index = random.randint(0, len(start) - 1)
        #保存当前正在被掩码的词汇
        word = sentence[start[index]:end[index]]
        mylist = list(sentence)
        j = start[index]
        #将词汇掩码为masked(♥)
        while j < end[index]:
            mylist[j] = masked
            j += 1
        sentence = ''.join(mylist)
        #start_index加入start[index]，为什么用这个不用append，因为要保证start小的在前面
        bisect.insort(start_index, start[index])
        insort_index = start_index.index(start[index])
        term_length.insert(insort_index, end[index] - start[index])
        sentence_masked_terms.insert(insort_index, word)
        #更新mask剩余数量
        mask -= (end[index] - start[index])
        start.pop(index)
        end.pop(index)
        if len(start) == 0:
            break
    random_int=[]
    #如果mask的掩码率没有达到标准，继续使用随即掩码
    #为什么不一边随机挑选index一边掩码，因为需要顺序，所以需要先挑选，再排序，再掩码
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
#根据设置的最长句子长度来进行分句
def split_sentences(sentence):
    MAX_LENGTH = 500
    sentences = []
    #分句的依据，多个中文结句符号
    sub_sentences=re.split(r'([。！!？?])',sentence)
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

#添加空白，在被掩码符号的前后加上空格，因为在做tokenizer的时候，♥[UNK]-->[UNK]   如果加上空格后 ♥ [UNK]-->♥[UNK]，有时候会将掩码符号和后面的不知名符号连接起来组成一个不知名符号
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
#词表文件，一行一个词
term_path=r'/home/sirui/WMM/Medicine/baike_data/WWM_terms.txt'
with open(term_path,'r',encoding='utf-8') as f:
    terms=f.readlines()
for i in range(len(terms)):
    terms[i]=terms[i].replace('\n','')
#去重
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
            #去除所有空格，因为不去除有时候会随机选择到空格进行掩码
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




