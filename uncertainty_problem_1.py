import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats

x = np.array([8.2, 8.5, 8.1, 8.4, 8.3])
n = len(x)
x_bar = np.mean(x)
C_ind = x_bar

dev = x - x_bar 
dev_sq = dev**2 

sum_dev = np.sum(dev) # sum of deviations
sum_dev_sq = np.sum(dev_sq) # sum of squares

s_sq = sum_dev_sq / (n - 1)
s = math.sqrt(s_sq)

# type A uncertainty

u2_A = s_sq / n
u_A = math.sqrt(u2_A)
v = n - 1

# type B uncertainty

# calibration - normal, U = 0.6, k = 2

U = 0.6
k = 2

del_cal = U / k
u2_del_cal = del_cal**2

# resolution - uniform, a = +-0.05 --> numeric integration

def uniform(a, N):
    dx = (2 * a) / N

    area = 0.0
    u2_del = 0.0

    def pdf(x):
        return 1 / (2 * a)

    for i in range(N):
        x = -a + (i + 0.5) * dx      
        area += pdf(x) * dx          
        u2_del += x**2 * pdf(x) * dx  

    u_del = math.sqrt(u2_del)

    return u_del, u2_del

uni_tup = uniform(0.05, 10000)
u_del_res = uni_tup[0]
u2_del_res = uni_tup[1]


# alignment - triangular, a = +-0.3 --> numeric integration

def triangular(a, N):
    dx = a / N

    def pdf(x):
        return (a - x) / a**2

    area = 0.0
    u2_del = 0.0
    for i in range(N):
        x = (i + 0.5) * dx           
        area += 2 * pdf(x) * dx     
        u2_del += 2 * x**2 * pdf(x) * dx   

    u_del = math.sqrt(u2_del)

    return u_del, u2_del

tri_tup = triangular(0.3, 10000)
u_del_align = tri_tup[0]
u2_del_align = tri_tup[1]

# sensitivity coefficient

def model(c_ind, d_cal, d_res, d_align):
    return c_ind + d_cal + d_res + d_align

h = 1e-6                        # tiny nudge
ref = [0.0, 0.0, 0.0, 0.0]      # reference point (any values work, see note)

def sensitivity(i):
    nudged = ref.copy()
    nudged[i] += h            
    return (model(*nudged) - model(*ref)) / h

c_A     = sensitivity(0)        
c_cal   = sensitivity(1)        
c_res   = sensitivity(2)       
c_align = sensitivity(3)        

# combined standard uncertainty

term_A     = c_A**2     * u2_A
term_cal   = c_cal**2   * u2_del_cal
term_res   = c_res**2   * u2_del_res
term_align = c_align**2 * u2_del_align

u2_c = term_A + term_cal + term_res + term_align
u_c  = math.sqrt(u2_c)

# effective degrees of freedom

nu_cal   = math.inf          # Type B → infinite degrees of freedom
nu_res   = math.inf
nu_align = math.inf

denominator = (term_A**2     / v        +
               term_cal**2   / nu_cal   +
               term_res**2   / nu_res   +
               term_align**2 / nu_align)

nu_eff = u_c**4 / denominator

# coverage factor and expanded uncertainty

t95   = stats.t.ppf(0.975, nu_eff)   # t-table value for 95 %, ν = ν_eff
k_cov = 2                            # conventional choice
U_exp = k_cov * u_c                  # expanded uncertainty, k = 2
U_196 = 1.96 * u_c                   # alternative, k = 1.96
U_rep = round(U_exp, 2)              # rounded: 0.67

# final result and interval

C_result = C_ind                    
lower = C_result - U_rep
upper = C_result + U_rep

print("PROBLEM 1 - GUM")
print("mean =", C_ind)
print("s =", s)
print("u_A =", u_A)
print("u_cal =", del_cal)
print("u_res =", u_del_res)
print("u_align =", u_del_align)
print("uc =", u_c)
print("nu_eff =", nu_eff)
print("k95 =", t95)
print("U(k = 2) =", U_exp)
print("U(k95) =", t95 * u_c)


# monte carlo simulation

M = 200000                         
rng = np.random.default_rng(2026)    

# samplers

def sample_uniform(a, b, size):
    U = rng.uniform(0, 1, size)
    return a + (b - a) * U

def sample_triangular(a, c, b, size):        # a = min, c = peak, b = max
    U = rng.uniform(0, 1, size)
    F = (c - a) / (b - a)
    left  = a + np.sqrt(U * (b - a) * (c - a))
    right = b - np.sqrt((1 - U) * (b - a) * (b - c))
    return np.where(U < F, left, right)

def sample_normal(mu, sigma, size):        
    U1 = 1 - rng.uniform(0, 1, size)        
    U2 = rng.uniform(0, 1, size)
    z = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    return mu + sigma * z

def sample_t(nu, size):                     
    z = sample_normal(0, 1, size)
    chi2 = np.zeros(size)
    for j in range(nu):
        chi2 += sample_normal(0, 1, size)**2  
    return z / np.sqrt(chi2 / nu)

C_ind_mc   = C_ind + (s / math.sqrt(n)) * sample_t(v, M)   # type A: t with ν = 4
d_cal_mc   = sample_normal(0, del_cal, M)                  # normal, mean 0, σ = 0.3
d_res_mc   = sample_uniform(-0.05, 0.05, M)                # uniform +-0.05
d_align_mc = sample_triangular(-0.3, 0, 0.3, M)            # triangular +-0.3, peak at 0

C_mc = model(C_ind_mc, d_cal_mc, d_res_mc, d_align_mc)

# plotting histogram

mean_mc = np.mean(C_mc)               # best estimate
u_mc    = np.std(C_mc, ddof=1)        # standard uncertainty
low_mc  = np.percentile(C_mc, 2.5)    # lower end of 95 % interval
high_mc = np.percentile(C_mc, 97.5)   # upper end of 95 % interval

plt.figure(figsize=(8, 5))
plt.hist(C_mc, bins=100, density=True, alpha=0.6, label="Monte Carlo")
plt.axvline(low_mc, color="k", linestyle="--", label="95% interval")
plt.axvline(high_mc, color="k", linestyle="--")
plt.xlabel("Cylindricity C (micro-meter)")
plt.ylabel("Probability density")
plt.title("Problem 1: Monte Carlo PDF")
plt.legend()

print('\n')

print("PROBLEM 1 - MONTE CARLO")
print("mean =", mean_mc)
print("u =", u_mc)
print("95% low =", low_mc)
print("95% high =", high_mc)

plt.show()
