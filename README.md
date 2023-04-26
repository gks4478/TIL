## Natural Languate Processing (NLP) for Beginners
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
