import sympy as sp

# Define the symbols
x = sp.symbols('x')

# Define the functions for each integral
f1 = sp.log(x)
f2 = sp.exp(-x**2)
f3 = 1 / (1 + x**2)

# Derivatives for the trapezoidal rule (second derivative)
f1_2nd_deriv = f1.diff(x, 2)
f2_2nd_deriv = f2.diff(x, 2)
f3_2nd_deriv = f3.diff(x, 2)

# Derivatives for the Simpson's rule (fourth derivative)
f1_4th_deriv = f1.diff(x, 4)
f2_4th_deriv = f2.diff(x, 4)
f3_4th_deriv = f3.diff(x, 4)

# Define the intervals and errors
intervals = [(1, 3), (-1, 1), (1/2, 5/2)]
errors = [10**-8, 10**-10, 10**-12]

# Compute the maximum of the absolute values of the derivatives over the intervals
# For trapezoidal rule
max_f1_2nd = max(abs(f1_2nd_deriv.subs(x, val)) for val in intervals[0])
max_f2_2nd = max(abs(f2_2nd_deriv.subs(x, val)) for val in intervals[1])
max_f3_2nd = max(abs(f3_2nd_deriv.subs(x, val)) for val in intervals[2])

# For Simpson's rule
max_f1_4th = max(abs(f1_4th_deriv.subs(x, val)) for val in intervals[0])
max_f2_4th = max(abs(f2_4th_deriv.subs(x, val)) for val in intervals[1])
max_f3_4th = max(abs(f3_4th_deriv.subs(x, val)) for val in intervals[2])

# Calculate n for each method and integral
# Trapezoidal rule formula: n^2 = (b - a)^3 * max|f''(x)| / (12*epsilon)
# Simpson's rule formula: n^4 = (b - a)^5 * max|f''''(x)| / (180*epsilon)

# For trapezoidal rule
n_trapezoidal = [
    ((intervals[0][1] - intervals[0][0])**3 * max_f1_2nd / (12 * errors[0]))**(1/2),
    ((intervals[1][1] - intervals[1][0])**3 * max_f2_2nd / (12 * errors[1]))**(1/2),
    ((intervals[2][1] - intervals[2][0])**3 * max_f3_2nd / (12 * errors[2]))**(1/2)
]

# For Simpson's rule
n_simpson = [
    ((intervals[0][1] - intervals[0][0])**5 * max_f1_4th / (90 * errors[0]))**(1/4),
    ((intervals[1][1] - intervals[1][0])**5 * max_f2_4th / (90 * errors[1]))**(1/4),
    ((intervals[2][1] - intervals[2][0])**5 * max_f3_4th / (90 * errors[2]))**(1/4)
]

print(n_trapezoidal, n_simpson)
