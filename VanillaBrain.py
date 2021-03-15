#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import math
from collections import Counter
from konlpy.tag import Mecab
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from soykeyword.proportion import CorpusbasedKeywordExtractor


# In[ ]:


class VanillaNews: # train : 훈련용 토큰화 파일 경로, model : 딥러닝 훈련 모델 경로
    def __init__(self, train = None, model = None, txt = None, recom_raw = None):
        
        # 훈련용 토큰화 파일 전처리
        train_frame = pd.read_csv(train, header = None)
        token_train = []
        for i in range(len(train_frame)):
            token = train_frame.loc[i, :].values
            token = token.tolist()
            j = -1
            for k in range(len(token)):
              j += 1
              if not isinstance(token[j], str) and math.isnan(token[j]):
                del token[j]
                j = j-1
            token_train.append(token)
        
        threshold = 3
        rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
        
        # 정수 인코딩
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(token_train) 
        total_cnt = len(tokenizer.word_index) # 단어의 수
        
        for key, value in tokenizer.word_counts.items():
            if(value < threshold):
                rare_cnt = rare_cnt + 1
        
        vocab_size = total_cnt - rare_cnt + 2
        self.tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
        self.tokenizer.fit_on_texts(token_train)
        
        self.mecab = Mecab()
        
        # 분류 모델에 필요한 딥러닝 모델, 변수
        self.deeprunning = load_model(model)
        self.category = 0
        
        # 추천 모델에 필요한 keyword
        with open(txt, 'r') as t:
            word = t.readlines()
        self.word = word[0].split()
        
        # 추천 모델에 필요한 data
        self.recom_data = pd.read_csv(recom_raw)
    
    def news_input(self, file = None): # file = 분류하거나 추천할 기사 내용
        
        # 명사 추출
        self.test_re = [m for m in self.mecab.nouns(file) if len(m) > 1]
        self.test_ca = [self.test_re]
        
    def news_category(self): # 뉴스 카테고리 분류
        
        # 숫자로 바꾸고
        test_category = self.tokenizer.texts_to_sequences(self.test_ca)
        
        # 패딩
        max_len = 500
        test_category = pad_sequences(test_category, maxlen = max_len)

        # 기사 카테고리 분류
        predict = self.deeprunning.predict(test_category)
        self.category = np.argmax(predict) # 분류 결과
        
        return self.category
    
    def news_recommend(self): # 뉴스 추천
        
        # test main keyword 검색
        keyword = []
        for n in self.test_re:
            if n in self.word:
                keyword.append(n)
        self.main = Counter(self.test_re).most_common()[0][0]
        
        # 추천 데이터 생성
        hang = []
        for o, p in enumerate(self.recom_data['Keyword']):
            if self.main in p:
                hang.append(o)
        recommend = self.recom_data.iloc[hang, :]
        # 최신별로 정렬
        recommend_sort = recommend.sort_values(by = 'Time', axis = 0,
                                               ascending = False)
        self.best = recommend_sort[:10]
        
        return self.best
    
    def first_user(self): # 초기 사용자 추천 모델, 오늘의 키워드
        
        self.today = self.recom_data[self.recom_data['Time'] == max(self.recom_data['Time'])]['News']
        
        # 키워드 생성
        corpusbased_extractor = CorpusbasedKeywordExtractor(
        min_tf=1,
        min_df=1,
        tokenize=self.mecab.nouns,
        verbose=True)
        corpusbased_extractor.train(self.today)
        
        # 키워드 전처리 및 추출
        stop = ['기자', '투데이', '머니', '사진']
        today_keyword = [i for i in list(corpusbased_extractor._tfs.items())
                         if len(i[0]) > 1 and i[0] not in stop]
        today_keyword = sorted(today_keyword, reverse = True,
                               key = lambda item: item[1])
        self.today_main = today_keyword[0][0]
        
        # 추천 데이터 생성
        documents = corpusbased_extractor.get_document_index(self.today_main)
        self.today_recom = self.today.reset_index().iloc[documents[:10], :]
        
        return self.today_recom

class VanillaNewsKeyword: # train : 훈련용 토큰화 파일 경로, model : 딥러닝 훈련 모델 경로
    def __init__(self, train = None, model = None, recom_raw = None):
        
        # 훈련용 토큰화 파일 전처리
        train_frame = pd.read_csv(train, header = None)
        token_train = []
        for i in range(len(train_frame)):
            token = train_frame.loc[i, :].values
            token = token.tolist()
            j = -1
            for k in range(len(token)): 
              j += 1
              if not isinstance(token[j], str) and math.isnan(token[j]):
                del token[j]
                j = j-1
            token_train.append(token)
        
        threshold = 3
        rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
        
        # 정수 인코딩
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(token_train) 
        total_cnt = len(tokenizer.word_index) # 단어의 수
        
        for key, value in tokenizer.word_counts.items():
            if(value < threshold):
                rare_cnt = rare_cnt + 1
        
        vocab_size = total_cnt - rare_cnt + 2
        self.tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
        self.tokenizer.fit_on_texts(token_train)
        
        self.mecab = Mecab()
        
        # 분류 모델에 필요한 딥러닝 모델, 변수
        self.deeprunning = load_model(model)
        self.category = 0
        
        # 추천 모델에 필요한 data
        self.recom_data = pd.read_csv(recom_raw)
    
    def news_input(self, file = None): # file = 분류하거나 추천할 기사 내용
        
        # 명사 추출
        self.test_re = [m for m in self.mecab.nouns(file) if len(m) > 1]
        self.test_ca = [self.test_re]
        
    def news_category(self): # 뉴스 카테고리 분류
        
        # 숫자로 바꾸고
        test_category = self.tokenizer.texts_to_sequences(self.test_ca)
        
        # 패딩
        max_len = 500
        test_category = pad_sequences(test_category, maxlen = max_len)

        # 기사 카테고리 분류
        predict = self.deeprunning.predict(test_category)
        self.category = np.argmax(predict) # 분류 결과
        
        return self.category
    
    def news_keyword(self): # 뉴스 키워드
        
        self.main = Counter(self.test_re).most_common()[0][0]
    
    def news_recommend(self): # 뉴스 추천
        
        # 추천 데이터 생성
        hang = []
        for o, p in enumerate(self.recom_data['Keyword']):
            if self.main in p:
                hang.append(o)
        recommend = self.recom_data.iloc[hang, :]
        # 최신별로 정렬
        recommend_sort = recommend.sort_values(by = 'Time', axis = 0,
                                               ascending = False)
        self.best = recommend_sort[:10]
        
        return self.best
    
    def first_user(self): # 초기 사용자 추천 모델, 오늘의 키워드
        
        self.today = self.recom_data[self.recom_data['Time'] == max(self.recom_data['Time'])]['News']
        
        # 키워드 생성
        corpusbased_extractor = CorpusbasedKeywordExtractor(
        min_tf=1,
        min_df=1,
        tokenize=self.mecab.nouns,
        verbose=True)
        corpusbased_extractor.train(self.today)
        
        # 키워드 전처리 및 추출
        stop = ['기자', '투데이', '머니', '사진']
        today_keyword = [i for i in list(corpusbased_extractor._tfs.items())
                         if len(i[0]) > 1 and i[0] not in stop]
        today_keyword = sorted(today_keyword, reverse = True,
                               key = lambda item: item[1])
        self.today_main = today_keyword[0][0]
        
        # 추천 데이터 생성
        documents = corpusbased_extractor.get_document_index(self.today_main)
        self.today_recom = self.today.reset_index().iloc[documents[:10], :]
        
        return self.today_recom

