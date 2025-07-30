import numpy as np

# 문제 1
x_values = np.array([1,2,3])
likelihood = np.array([0.2,0.5,0.3])
exp_x = np.sum(x_values*likelihood)
print(exp_x)

# 문제 2
var_x = np.sum((x_values-exp_x)**2*likelihood)
var_x

# 문제 3
new_values = x_values*2+3
new_exp_x = np.sum(new_values*likelihood)
new_exp_x

# 문제 4
new_var = np.sum((new_values-new_exp_x)**2*likelihood)
new_var

# 문제 5
x_values = np.array([0,1,2,3])
likelihood = np.array([0.1,0.3,0.4,0.2])
exp_x = np.sum(x_values*likelihood)
var_x = np.sum((x_values-exp_x)**2*likelihood)
print("기대값=",exp_x,"분산=",var_x)

# 문제 8
exp_x = 5
exp_y = 3
ans = 2*exp_x-exp_y+4
print(ans)

# 문제 9
# aX+b의 기대값 = au + b
# aX+b의 분산 = a^2 * 시그마


# 문제 10
p = 1-0.3-0.4
p
x_values = np.array([1,2,3])
likelihood = np.array([0.3,p,0.4])
exp_x = np.sum(x_values*likelihood)
print(exp_x)

# 문제 11
x_values = np.array([1,2,4])
likelihood = np.array([0.2,0.5,0.3])
exp_x = np.sum(x_values*likelihood)
exp_x2 = np.sum(x_values**2*likelihood)
var_x = np.sum((x_values-exp_x)**2*likelihood)

print("E(X) = ", exp_x, " E(X^2) =", exp_x2, "Var(X) =", var_x)
# 해당 값이 분산과 같음
exp_x2 - exp_x**2
