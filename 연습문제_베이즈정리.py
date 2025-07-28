import numpy as np

# 문제 1 – 회사 직원의 건강 문제와 흡연 확률
# 한 회사에서 무작위로 선택된 직원이 건강 문제가 있을 확률은 0.25입니다. 건강 문제가 있는 직원은 건강 문제가 없는 직원보다 흡연자일 확률이 두 배 높습니다.
# 직원이 흡연자라는 사실을 알았을 때, 그가 건강 문제를 가지고 있을 확률을 계산하십시오.
0.4

# 문제 2
prior = np.array([0.16,0.18,0.20])
likelihood = np.array([0.05,0.02,0.03])
p_break = np.sum(prior*likelihood)
posterior = (prior*likelihood) / p_break
posterior
# 정답
ans = round(posterior[0],2)
print(ans)