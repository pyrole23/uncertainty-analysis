import numpy as np
import math
import matplotlib.pyplot as plt

def model(m, P, V):
    C_zn = (1000 * m * P) / V
    return C_zn

# best estimates

P = np.array([0.9985, 1.0005])
V = np.array([149.6, 150.4])
P_not = np.mean(P)
m_not = 300.75
V_not = 150
N = 10000

C_zn_not = model(m_not, P_not, V_not)

# type B uncertainties

# purity --> uniform (max, min)

def uniform(x_min, x_max, N):
    a_x = (x_max - x_min) / 2
    dx = (2 * a_x) / N 

    def pdf(x):
        return 1 / (2 * a_x)

    area_x = 0.0
    u2_x = 0.0
    for i in range(N):
        x = -a_x + (i + 0.5) * dx          
        area_x += pdf(x) * dx               
        u2_x += x**2 * pdf(x) * dx        

    u_x = math.sqrt(u2_x)

    return u_x, u2_x

P_tuple = uniform(P[0], P[1], N)
u_P = P_tuple[0]
u2_P = P_tuple[1]

# mass --> gaussian(mean, SD)

m_sd = 0.009
u_m = 0.009
u2_m = 0.009**2

# volume --> triangular, repeatability, temperature

# filling --> triangular (min, max, min)

def triangular(x_max, x_min, N):
    a_x = (x_max - x_min) / 2
    dx = a_x / N

    def pdf(x):
        return (a_x - x) / a_x**2 

    area_x = 0.0
    u2_x = 0.0
    for i in range(N):
        x = (i + 0.5) * dx
        area_x += 2 * pdf(x) * dx            
        u2_x += 2 * x**2 * pdf(x) * dx

    u_x = math.sqrt(u2_x)
    return u_x, u2_x

V_tuple_fill = triangular(V[1], V[0], N)
u_V_fill = V_tuple_fill[0]
u2_V_fill = V_tuple_fill[1]

# repeatability --> (range, mean)

def rectangular(range_, N):
    x_rep = range_ / 2
    dx = (2 * x_rep) / N

    def pdf(x):
        return 1 / (2 * x_rep) 

    area_x = 0.0
    u2_x = 0.0

    for i in range(N):
        x = -x_rep + (i + 0.5) * dx           
        area_x += pdf(x) * dx               
        u2_x += x**2 * pdf(x) * dx        

    u_x = math.sqrt(u2_x)
    return u_x, u2_x

range_repeat = 0.09
V_tuple_repeat = rectangular(range_repeat, N)
u_V_repeat = V_tuple_repeat[0]
u2_V_repeat = V_tuple_repeat[1]

# temperature --> normal, at 95%, k = 1.96

u_temp = 0.1
u_V_temp = u_temp / 1.96
u2_V_temp = u_V_temp**2

# combined uncertainty

u2_V = u2_V_fill + u2_V_repeat + u2_V_temp
u_V = math.sqrt(u2_V)

# sensitivity coefficients

def sensitivity(f, inputs, i, h = 1e-4):
    up = list(inputs)
    down = list(inputs)
    up[i] += h
    down[i] -= h
    return (f(*up) - f(*down)) / (2 * h)

inputs = [m_not, P_not, V_not]

c_m = sensitivity(model, inputs, 0)
c_P = sensitivity(model, inputs, 1)
c_V = sensitivity(model, inputs, 2)

# individual contribution

m_contri = c_m * u_m
P_contri = c_P * u_P
V_contri = c_V * u_V
m2_contri = u2_m * c_m**2
P2_contri = u2_P * c_P**2
V2_contri = u2_V * c_V**2

# combined standard uncertainty

u2_combined = m2_contri + P2_contri + V2_contri
u_combined = math.sqrt(u2_combined)

print("PROBLEM 2 - GUM")
print("c_Zn (mg/L) =", C_zn_not)
print("u(P) =", u_P)
print("u(m) (mg) =", u_m)
print("u(V) (mL) =", u_V)
print("c_m =", c_m)
print("c_P =", c_P)
print("c_V =", c_V)
print("u_c (mg/L) =", u_combined)
print("U, k=2 (mg/L) =", 2 * u_combined)

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

P_mc      = sample_uniform(P[0], P[1], M)            # purity: uniform
m_mc      = sample_normal(m_not, u_m, M)                    # mass: gaussian
V_fill_mc = sample_triangular(V[0], V_not, V[1], M)    # filling: triangular
V_rep_mc = sample_uniform(-range_repeat / 2, range_repeat / 2, M)            # repeatability: rectangular
V_temp_mc = sample_normal(0, u_V_temp, M)                  # temperature: normal

V_mc = V_fill_mc + V_rep_mc + V_temp_mc
c_mc = model(m_mc, P_mc, V_mc)

mean_mc = np.mean(c_mc)
u_mc    = np.std(c_mc, ddof=1)
low_mc  = np.percentile(c_mc, 2.5)
high_mc = np.percentile(c_mc, 97.5)

plt.figure(figsize=(8, 5))
plt.hist(c_mc, bins=100, density=True, alpha=0.6, label="Monte Carlo")

plt.axvline(low_mc, color="k", linestyle="--", label="95% interval")
plt.axvline(high_mc, color="k", linestyle="--")
# plt.axvline(mean_mc, color="g", label="MC mean")
plt.xlabel("Zinc concentration c_Zn (mg/L)")
plt.ylabel("Probability density")
plt.title("Problem 2: Monte Carlo PDF")
plt.legend()

print('\n')

print("PROBLEM 2 - MONTE CARLO")
print("mean =", mean_mc)
print("u =", u_mc)
print("95% low =", low_mc)
print("95% high =", high_mc)

plt.show()