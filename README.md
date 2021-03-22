# News-Classification-Recommend
* 목표 : 뉴스 데이터를 이용하여 뉴스 카테고리를 자동으로 분류하는 모델 및 사용자가 읽은 기사와 유사한 키워드를 가진 기사를 추천하는 모델 생성 

## Problem
뉴스 추천 모델을 만들기 위해서 사용자 로그 데이터와 사용자들에게 추천해 줄 뉴스 데이터가 필요합니다. 하지만 뉴스 데이터 중 분류가 제대로 되지 않은 데이터가 절반 가량 됩니다. 이러한 이유에서 뉴스 분류 모델을 만들어 뉴스 데이터를 카테고리에 맞게 분류하고자 한다.
또한 뉴스 추천 모델을 만들어 사용자들의 취향에 맞추어 뉴스를 서비스 하고 장기적으로 사용자를 늘릴 수 있게 하려한다.

## Variable
X_train : 토큰화 뉴스 데이터
best_model : 딥러닝 모델
keyword_mecab : soykeyword로 추출한 키워드 목록
news_recommend : 추천해줄 데이터가 담긴 뉴스 데이터
(news_df_02100201) : (테스트용 뉴스 데이터)

## Accuracy (classification)
80%

## Mecab 설치
1. automake install
2. bash <(curl -s [https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh](https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh))

## Class
VanillaNews : Normal
VanillaNewsKeyword : + news_keyword(키워드 추출 따로)

## Time (기사 12716자 기준)
* init : 4분 11초
* news_input : 42.5ms
* news_category : 433ms
* news_recommend : 97.5ms

## Requires
* Tensorflow 2.0
* konlpy(Mecab)
* soykeyword

## Model
* Autoencoder

## References
