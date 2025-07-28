import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


# 문제 1 – 회사 직원의 건강 문제와 흡연 확률
# 한 회사에서 무작위로 선택된 직원이 건강 문제가 있을 확률은 0.25입니다. 건강 문제가 있는 직원은 건강 문제가 없는 직원보다 흡연자일 확률이 두 배 높습니다.
# 직원이 흡연자라는 사실을 알았을 때, 그가 건강 문제를 가지고 있을 확률을 계산하십시오.
prior = np.array([0.25,0.75])
likelihood = np.array([2,1])
p_break = np.sum(prior*likelihood)
p_break

posterior = (prior*likelihood) / p_break
posterior

# 문제 2
prior = np.array([0.16,0.18,0.20]) / 0.54 # 0.54에서 3가지 중 하나이니까
likelihood = np.array([0.05,0.02,0.03])
p_break = np.sum(prior*likelihood)
posterior = (prior*likelihood) / p_break
posterior
# 정답
ans = round(posterior[0],2)
print(ans)

# 문제 3









# 동전 두개 던질 때 확률분포표 시각화

x_array = np.array([0,1,2])
likelihood = np.array([0.25,0.5,0.25])

# 시각화
plt.figure(figsize=(6, 4))
plt.bar(x_array, likelihood, width=0.4, color='skyblue', edgecolor='black')

# 라벨과 타이틀
plt.xlabel('x 값')
plt.ylabel('확률 (P(x))')
plt.title('확률분포표 시각화')
plt.xticks(x_array)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 확률 값 표시
for x, p in zip(x_array, likelihood):
    plt.text(x, p + 0.02, f'{p:.2f}', ha='center')

plt.tight_layout()
plt.show()

