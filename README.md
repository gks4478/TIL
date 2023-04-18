### Titanic Top 4% with ensemble modeling
https://www.kaggle.com/code/yassineghouzam/titanic-top-4-with-ensemble-modeling/notebook
> #### 1. 데이터셋에서 결측기 인덱스 가져오기
```python
index_NaN_age= list(dataset['Age'][dataset['Age'].isnull()].index)
# dataset['Age'] 여기가 있어야 인덱스 가져오기 가능
```

> #### 2. 결측치를 특정 특성(열)의 값이 같은 행의 값으로 채우기
```python
# Age의 값이 결측치일 때 Pclass, Parch, SibSp의 값이 같은 행들을 찾아서 
# 그 행의 Age의 중간값으로 채우는데 만약 그게 NaN이라면 age_med로 채운다.
for i in index_NaN_age:
    age_med= dataset['Age'].median()
    age_pred= dataset['Age'][((dataset['SibSp'] == dataset.iloc[i]['SibSp']) &
                              (dataset['Parch'] == dataset.iloc[i]['Parch']) &
                              (dataset['Pclass']== dataset.iloc[i]['Pclass']))].median()
    if not np.isnan(age_pred):
        dataset['Age'].iloc[i]= age_pred
    else:
        dataset['Age'].iloc[i]= age_med
```

> ### 3. 어떤 리스트를 시리즈로 저장해서 데이터 프레임의 새로운 열에 저장하기
```python
dataset_title= [i.split(',')[1].split('.')[0].strip() for i in dataset['Name']] # 이름에서 호칭만 가져오기(Mr같은)
dataset['Title']= pd.Series(dataset_title)
```

> ### 4. 명목형 변수의 더미 dummies()
>* 범주형 변수: 명목형 변수와 서열형 변수가 있는데 명목형 변수는 순서가 없는 것이다.(color의 red, green, blue 와 같은)
>* 더미를 하는 이유는 모델링을 위한 것이다.(OneHotEncoder랑 동작은 비슷함)
>* 범주형 변수의 모든 카테고리를 각각 새로운 이진 변수 (0,1)로 변환한다.
```python
dataset= pd.get_dummies(dataset, columns= ['Embarked'], prefix= 'Em')
# prefix: 이름 지정 (원래가 apple이였다면 Em_apple로 변경됨)
```

> ### 5. 여러 머신러닝 모델 비교하기
>* 분류모델로 대표적인 SVC, Decision Tree, AdaBoost, Random Forest, Extra Trees, Gradient Boosting, Multiple layer perceprton (neural network), KNN, Logistic regression, Linear Discriminant Analysis 10개
>* 어떤 모델을 선택할지 결정 후 하이퍼파라미터 튜닝을 진행하는 방식
```python
# 1. kfold 교차 검증 생성
kfold = StratifiedKFold(n_splits=10) # n_splits: 10개로 나눈다.

# 2. 모델 생성
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

# 3. 각 모델별로 교차 검증 후 cv_results에 저장
cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4)) # cv: 교차검증 넣기, n_jobs: cpu 몇 개 쓸건가    

# 4. 평균과 표준편차를 구하고 저장한다.
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

# 5. 데이터 프레임 생성 후 그래프 출력
cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
```
> ![스크린샷 2023-04-19 045645](https://user-images.githubusercontent.com/77867734/232891002-7793203c-5e8f-4918-9fed-ad41129e2840.png)

> ### 6. 과적합이 났는지 그래프로 확인하기(이게 PDP 그건가?)
```python
# 그래프 그리는 함수
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# 적용하고자 하는 모델 넣기
# (모델 최고 하이퍼파라미터, 제목, X_train, Y_train, cv= 교차검증)
g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",X_train,Y_train,cv=kfold)
```
> ![스크린샷 2023-04-19 051936](https://user-images.githubusercontent.com/77867734/232895743-664fc7e5-fcc6-4d4e-a7ea-4d04d9bd026f.png)

> ### 7. 앙상블: VotingClassifier
>* 관련있는 모델을 섞는다. 여기서는 모델들끼리도 상관관계(Correlation)를 측정했다. 
```python
# VotingClassifier 앙상블
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, Y_train)
```
```python
# voting 후 최종 test set으로 예측하고 csv 파일로 저장하기
test_Survived = pd.Series(votingC.predict(test), name="Survived")
results = pd.concat([IDtest,test_Survived],axis=1)
results.to_csv("ensemble_python_voting.csv",index=False)
```
