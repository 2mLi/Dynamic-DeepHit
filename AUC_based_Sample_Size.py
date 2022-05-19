# this is the sample size calculator
# 


import scipy.stats as st
import sys
import math


# example sys.argv: 
# sys.argv = ['script.py', 'AUC', 'd', 'alpha', 'prev', 'method']

AUC = float(sys.argv[1])
d = float(sys.argv[2])
alpha = float(sys.argv[3])
prev = float(sys.argv[4])
method = sys.argv[5]

if AUC > 1 or AUC < 0: 
    print('Error: AUC should be within the range of [0, 1]. Ideally, AUC is better to be larger than 0.5. ')

# calculate needed parameters
phi_inverse_AUC = st.norm.cdf(AUC)
a = phi_inverse_AUC * 1.414

z_alpha_half = st.norm.ppf(1 - alpha / 2)

VAUC = (0.0099 * math.exp(-(a ** 2)/2)) * (6 * (a ** 2) + 16)



if method == 'sensitivity': 
    n = (z_alpha_half ** 2) * AUC * (1 - AUC) / ((d ** 2) * prev)
elif method == 'specificity': 
    n = (z_alpha_half ** 2) * AUC * (1 - AUC) / ((d ** 2) * (1 -prev))
elif method == 'AUC': 
    n = (z_alpha_half ** 2) * VAUC / (d ** 2)
print('Sample size = ', str(n))