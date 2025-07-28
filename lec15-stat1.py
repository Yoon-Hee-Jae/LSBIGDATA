import numpy as np

# 1~10까지 랜덤 숫자 5개 뽑기
samples = np.random.choice(np.arange(1, 11), size=5, replace=False)
# repalce=False 복원추출 여부
samples

# 동전 앞뒤 나오는 사건
outcome = np.random.choice(['H', 'T'])
print(f"결과: {outcome}")

# 수니 젤리 뭉이 문제에서 한번 더 접시가 깨질 확률
suni = 0.294
jelly = 0.354
mungy = 0.354

new = suni * 0.01 + jelly * 0.02 + mungy * 0.03

(suni * 0.01) / (new) # 0.142
(jelly * 0.02) / (new) # 0.343
(mungy * 0.03) / (new) # 0.515

# numpy 풀이 방식
prior = np.array([0.5,0.3,0.2])
likelihood = np.array([0.01,0.02,0.03])
p_break = np.sum(prior*likelihood)
posterior = (prior*likelihood) / p_break
posterior

# 접시가 한번 더 깨질 경우
prior = posterior
p_break = np.sum(prior*likelihood)
posterior = (prior*likelihood) / p_break
posterior

