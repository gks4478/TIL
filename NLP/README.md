## 2. NLP using GloVe Embeddings(99.87% Accuracy)
https://www.kaggle.com/code/madz2000/nlp-using-glove-embeddings-99-87-accuracy
### 1. beatuifulsoup
```python
from bs4 import BeautifulSoup

def strip_html(text):
    soup= BeautifulSoup(text, 'html.parser')
    # text를 파싱하는데 html.parser를 사용한다.
    return soup.get_text()
    # text 파싱한거에서 html의 모든 태그는 제거하고 순수 텍스트만 반환 한다.
# parser: 문장의 구조 분석, 오류 점검 프로그램
```

### 2. wordcloud
```python
from wordcloud import WordCloud, STOPWORDS

plt.figure(figsize= (20, 20))
wc= WordCloud(background_color= 'white', max_words= 100, width= 1600, height= 800, stopwords= STOPWORDS).generate(' '.join(df[df.label== 1].text))
# max_words: 생성될 최대 단어 수
# stopwords: stopwords 지정, 안하면 라이브러리에서 제공하는 기본값
# ''.join(df[df.label== 1].text): 데이터 프레임 df에서 label 값이 1인 행들의 text열
# join 함수를 이용해서 선택된 text 열의 모든 행을 하나의 문자열로 합친다.
# generate: join된 문자열로 단어 구름을 생성하고 WordCloud 객체에 저장
plt.imshow(wc, interpolation= 'bilinear')
plt.axis('off')
# 그래프의 축을 끈다.
plt.show()
```

### 3. .Tokenizer
```python
from keras.preprocessing import text, sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer= text.Tokenizer(num_words= max_features)
# 1. 문장의 단어집합을 생성한다.
# 2. 각 단어의 고유한 정수 인덱스를 부여한다.
# 3. 단어집합을 토큰화하고 각 단어에 해당하는 인덱스로 매핑한다
# 함수이다.

tokenizer.fit_on_texts(x_train)
# x_train으로 tokenizer 함수를 실행한다.

tokenized_train= tokenizer.texts_to_sequences(x_train)
# 단어의 인덱스를 정수로 변환한다.

x_train= pad_sequences(tokenized_train, maxlen= maxlen)
# 시퀀스의 길이를 동일하게 맞춘다.
# tokenized_train이 maxlen 보다 길면 maxlen에 맞춰서 자른다. 짧으면 0으로 채운다.
```

### 4. GloVe
>* LSA, Word2Vec의 장점을 모두 활용한다.
>* 모든 단어 쌍을 분석해서 단어들 간의 동시 출현 정보를 기반으로 단어 벡터를 학습한다.
>* 임베딩 된 '중심 단어 벡터와 주변 단어 벡터의 내적'이 전체 코스피에서의 '동시 등장 확률'이 되도록 한다.
>
> ![1](https://user-images.githubusercontent.com/77867734/235769801-e5dd633f-e328-4073-9b1c-57984913bb05.png)
>
>* https://warm-uk.tistory.com/11

### 5. Embedding
```python
model.add(Embedding(max_features, output_dim= embed_size, weights= [embedding_matrix], 
                    input_length= maxlen, trainable= False))
# max_features: 총 단어 개수
# output_dim: 임베딩 벡터의 크기(임베딩 출력 차원)
# weights: 미리 학습된 벡터 초기값
# input_length: 입력 시퀀스 길이
# trainable: 가중치 학습할지의 여부, False이면 미리 학습된 임베딩 벡터 고정(weights 값 고정)

model.add(LSTM(units= 128, return_sequences= True, recurrent_dropout= 0.25, dropout= 0.25))
model.add(LSTM(units= 64, recurrent_dropout= 0.1, dropout= 0.1))
# LSTM: 일반적으로 처리 작업에 사용된다. 입력 시퀀스를 처리하는 동안 이전 상태를 보존할 수 있으므로 이전 정보가 현재 출력에 반영되어 일부 정보가 손실되지 않는다.
# units: LSTM의 출력공간의 차원을 나타낸다.
# return_sequences: recurrent_activation 함수를 계산하기 전에 dropout을 적용하는 비율(과적합 방지에 사용)
# dropout: 출력에 대한 dropout 비율(일반화 성능을 향상시키기 위해 사용)

model.add(Dense(units= 32, activation= 'relu'))
# LSTM의 출력을 입력으로 받아 중간 특성을 추출하기 위한 것(이전 LSTM 레이어의 출력을 32차원으로 매핑한다.)
model.add(Dense(1, activation= 'sigmoid'))
# 이진 분류 문제이라서 차원은 1이어야 한다.

model.compile(optimizer= keras.optimizers.Adam(lr= 0.01), loss= 'binary_crossentropy', metrics= ['accuracy'])
# optimizer: 모델의 가중치를 업데이트하는 방법을 결정하는 알고리즘
# lr: 학습률
# loss: 손실함수 결정(이진분류는 binary_crossentropy 사용)
# metrics: 모델이 평가하는 지표 결정
```

### 6. 콜백callback함수: ReduceLROnPlateau
```python
learning_rate_reduction= ReduceLROnPlateau(monitor= 'val_accuracy', patience= 2, varbose= 1,
                                           factor= 0.5, min_lr= 0.00001)
# 검증 손실이 개선 되지 않을 때(= patience) 학습률을 동적으로 감소시켜 학습을 더욱 세밀하게 조정한다.
# monitor: 모니터링할 지표 지정(여기서는 검증 정확도를 설정한다.
# patience: 검증손실이 2번 연속 개선되지 않으면 학습률을 감소시킨다.
# factor: 학습률을 감소시킬 비율
# min_lr: 학습률의 하한구성, 학습률이 min_lr보다 작아지면 더 이상 학습률을 감소시키지 않는다.
```

### 7. 모델 학습
```python
history= model.fit(x_train, y_train, batch_size= batch_size, validation_data= (x_test, y_test), 
                    epochs= epochs, callbacks= [learning_rate_reduction])
# batch_size: 한번에 처리할 데이터 샘플 개수
# validation_data= (x_test, y_test): x_test와 y_test을 튜플 형태로 전달한다.
# epochs: 학습을 반복할 횟수이다.
# callbacks: 학습 도중에 사용할 콜백함수 리스트이다.
```

---
## 1. Natural Languate Processing (NLP) for Beginners
#### 1. CountVectorizer
>* 각 문서에 어떤 단어가 몇 번 등장했는지를 파악할 때 사용한다.
>* 각 단어를 세서 문서를 벡터화 한다.
>
> ![스크린샷 2023-04-26 221853](https://user-images.githubusercontent.com/77867734/234592424-d171e7f0-42c4-4af8-926d-4f8ef895b61c.png)
>
> https://radish-greens.tistory.com/3
```python
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# train, test 예제
simple_train = ['call you tonight', 'Call me a cab', 'Please call me... PLEASE!']
simple_test = ["please don't call me"]

vect= CountVectorizer()
simple_train_dtm= vect.fit_transform(simple_train) # train 학습
simple_test_dtm= vect.transform(simple_test) # test 학습
# fit: 데이터를 모델에 학습 시킨다.
# transform: fit을 기반으로 데이터를 변환한다.
# fit_transform: fit과 transform을 합친것(test set에서는 사용하면 안된다. 학습과 변환이 동시에 일어나기 때문)

print('<simple_train>')
display(pd.DataFrame(simple_train_dtm.toarray(), columns= vect.get_feature_names_out()))
print('<simple_test>')
display(pd.DataFrame(simple_test_dtm.toarray(), columns= vect.get_feature_names_out()))
# toarray(): 벡터화된 문서 출력
# .get_feature_names_out(): 단어장 출력
```
> ![스크린샷 2023-04-26 232538](https://user-images.githubusercontent.com/77867734/234606913-e5569ea4-11a1-4996-b1a0-4720a7352395.png)

#### 2. TF-IDF
>* TF(Term Frequency): 단어 빈도
>* DF(Document Frequency): 문서에서 특정 단어가 등장한 문서의 개수
>* 예: A,B,C,D의 문서가 있다 특정 단어 t가 등장하는 문서는 A,B이다. A에는 t가 100번, B에는 t가 1000번 이 나온다. 하지만 이는 문서를 기준으로 하기 때문에 2이다.
>* IDF(Inverse Documnet Frequency): DF와 역수 관계이다.
>* TF-IDF: 전체 문서에서 자주 나오는 단어의 중요도는 낮다고 판단되고 특정 문서에만 나오는 단어는 중요도가 높다고 판단한다.
>* https://heytech.tistory.com/337

#### 3. .predict_proba()
>* 분류 모델에서 클래스별로 확률값을 알려준다
>* 결과값이 [클래스1, 클래스2] 인데 각각 클래스1에 대한 확률값, 클래스2에 대한 확률값이다.

#### 4. MultinomialNB()
>* 이산기능(예: 텍스트 분류를 위한 단어 수)이 있는 분류에 적합하다.
