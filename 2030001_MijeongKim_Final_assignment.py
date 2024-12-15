# <기업의 광고 및 홍보 효과 예측> 2030001 김미정

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# 광고 및 홍보 효과 예측을 위한 n개의 표본
n = 100

# 기업의 광고 및 홍보 효과에 영향을 미치는 요인들을 리스트 형식으로 정의
money = [] # 광고에 투자한 비용
days = [] # 광고 노출 기간
views = [] # 광고가 노출된 횟수
others_money = [] # 다른 경쟁 기업의 광고 투자 비용

# 예측할 광고 효과를 리스트 형식으로 정의
ad_effect = []

np.random.seed(1) # 항상 같은 난수 값을 생성하기 위해 괄호 안에 임의의 값 1을 입력 (재현 가능한 난수 생성 방법을 chatgpt에 물어보았습니다)

# n개의 표본을 각각의 요인에 해당하는 리스트에 저장
# np.random.normal(평균, 표준편차): 정규 분포를 따르는 난수 생성 함수
for i in range(n):
    money.append(np.random.normal(5000, 1000)) # 광고에 투자한 비용
    days.append(np.random.normal(20, 5)) # 광고 노출 기간
    views.append(np.random.normal(626262, 62000)) # 광고가 노출된 횟수
    others_money.append(np.random.normal(4000, 1000)) # 경쟁 기업의 광고 투자 비용

# 각각의 요인을 모두 고려한 회귀를 완성해야 하므로, for문을 이용하여 각각의 요인들을 모두 한 번에 결합한 배열 만들기
X = [] # 각 요인들을 결합할, 비어있는 배열 X 정의

for i in range(n):
    X.append([money[i], days[i], views[i], others_money[i]]) # type error가 발생하여, 이에 대한 해결 방법을 찾지 못하여 chatgpt에게 type error를 고칠 수 있는 방법에 대해 물어보고, 하나의 리스트로 묶어서 append 하는 방식으로 코드를 수정하였습니다.

# 배열에 대한 연산을 간단히 하기 위해 리스트를 NumPy 배열로 변환
X = np.array(X)
money = np.array(money)
days = np.array(days)
views = np.array(views)
others_money = np.array(others_money)

# 각 요인별 가중치를 임의로 설정하여 광고 효과에 미치는 영향을 ad_effect로 저장 (ad_effect는 광고 효과를 나타내는 척도)
ad_effect = 0.4*money + 0.3*days + 0.3*views - 0.3*others_money


# 광고 효과에 랜덤 노이즈를 만들어주기
noise = [] # 랜덤 노이즈를 리스트 형태로 정의

np.random.seed(1)

# 정규분포를 따르는 n개의 랜덤 노이즈를 noise 리스트에 저장
for i in range(n):
    noise.append(np.random.normal(0, 100))

# 광고 효과에 랜덤 노이즈를 더해주기
ad_effect = ad_effect + np.array(noise)
y = ad_effect


# 데이터의 순서를 섞기 (Shuffle)
i = np.random.permutation(n)
shuffle_X = X[i]
shuffle_y = y[i]


# 3:1의 비율로 train과 test data를 나눠준다
N = int(n * (3/4))

# train data 분리
X_train = shuffle_X[:N]
y_train = shuffle_y[:N]

# test data 분리
X_test = shuffle_X[N:]
y_test = shuffle_y[N:]

# linear regression을 학습하기
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

lin_Predict_effect = lin_reg.predict(X_test) # test data에 대해 광고 효과 예측

# knn regression을 학습하기
knn_reg = KNeighborsRegressor(n_neighbors=4)
knn_reg.fit(X_train, y_train)

knn_Predict_effect = knn_reg.predict(X_test) # test data에 대해 광고 효과 예측

# R^(결정 계수)를 계산하여 linear regression, knn regression 성능 평가
r2_lin_reg = lin_reg.score(X_test, y_test)
r2_knn_reg = knn_reg.score(X_test, y_test)

print('Linear regression R^2: ', r2_lin_reg)
print('knn regression R^2: ', r2_knn_reg)


# linear regression, knn regression 모델을 이용한 광고 효과 예측 결과 plot하기
plt.figure(figsize=(8, 6)) # figure의 크기를 조작하는 방법을 chatgpt로 물어보았습니다.
plt.scatter(y_test, lin_Predict_effect, label='Linear Regression', color='red') # linear regression으로 예측한 광고 효과
plt.scatter(y_test, knn_Predict_effect, label='knn Regression', color='blue') # knn regression으로 예측한 광고 효과
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--g', label="Ideal Prediction") # y_test의 min, max값을 통해 이상적인 선(예측값=실제값인 경우의 선)을 그려, 실제값과 예측값이 일치하는 정도를 시각화한다.

plt.legend()
plt.title('Predicted Effect of Advertisement & Promotion')
plt.xlabel('Actual Effect')
plt.ylabel('Predicted Effect')
plt.show()


# 스스로에게 부여하는 학점: A+
# 그렇게 생각한 이유:
# 저는 전산물리학을 수강하기 이전에, 이번 학기 초까지만 해도 파이썬 언어의 아주 기초 (간단한 변수, 반복문, 연산 등)에 대해 부분적인 지식만 있는 수준이었습니다.
# 그런데, 이번 전산물리학 과목을 수강하면서, 파이썬을 이용한 다양한 명령문들을 스스로 작성할 수 있게 되었을 뿐만 아니라
# 모듈을 이용하여 그래프 그리기, 데이터 피팅과 미분 및 적분을 직접 코드를 작성하여 실행할 수 있게 되었으며,
# 2차원 포물체의 운동 및 로트카-볼테라 방정식 등을 풀 수 있는 메커니즘을 이해할 수 있게 되었습니다.
# KNN, linear/polynomial/logistic regression에 대한 개념을 처음 접하였는데, 이를 이해하고 직접 파이썬 코드로 구현하는 방법을 익히기 위해
# 스스로 연습해보려는 노력을 많이 하였고, 그 결과 마지막으로 기말 과제에서 linear regression과 knn regression을 이용하여 '기업의 광고 및 홍보 효과 예측'을 하는 코드를 스스로 직접 구현해볼 수 있었습니다.
# 이전에는 파이썬 코딩에 대한 자신감도 없었고 제 스스로 전산물리에 대해 아는 것이 거의 없다고 생각하였는데, 학기가 끝나가는 현재에는 아주 많이 성장하였다고 느껴집니다.
# 머신러닝이 무엇인지 전혀 알지 못했는데, 강의를 수강하면서 물리와 코딩, 머신러닝과 우리 생활이 어떻게 연관이 되는지, 어떻게 적용이 될 수 있는지 깨닫게 되었습니다. 
# 수업시간에는 배우는 새로운 내용들을 이해해보기 위해 스스로 정말 많이 노력하였고, 반드시 코드를 직접 쳐보면서 고민하다가 이해가 되지 않는 부분은 교수님께 질문을 드리기도 하며
# 파이썬 문법부터 머신러닝까지 최대한 이해하기 위해 노력해왔습니다.
# 이러한 노력 끝에 완성한 기말 과제를 통해 매우 큰 보람을을 느꼈고 이번 학기에 특히나 기억에 남는 과목이라고 생각합니다. 감사합니다.