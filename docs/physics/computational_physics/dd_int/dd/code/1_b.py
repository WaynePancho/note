import sympy as sp

# Define the function and the variable
x = sp.symbols('x')
f = sp.exp(x) / (sp.sin(x)**3 + sp.cos(x)**3)

h = 1/128

# Two-point formula for the first derivative
f_prime_2p = (f.subs(x, x + h) - f.subs(x, x - h)) / (2 * h)

# Three-point formula for the first derivative
f_prime_3p = (-3 * f.subs(x, x) + 4 * f.subs(x, x + h) - f.subs(x, x + 2 * h)) / (2 * h)

# Five-point formula for the first derivative
f_prime_5p = (-25 * f.subs(x, x) + 48 * f.subs(x, x + h) - 36 * f.subs(x, x + 2 * h) + 16 * f.subs(x, x + 3 * h) - 3 * f.subs(x, x + 4 * h)) / (12 * h)

# Evaluate the first derivatives at x = 0
f_prime_2p_at_0 = f_prime_2p.subs(x, 0).evalf()
f_prime_3p_at_0 = f_prime_3p.subs(x, 0).evalf()
f_prime_5p_at_0 = f_prime_5p.subs(x, 0).evalf()

f_prime_2p_at_0, f_prime_3p_at_0, f_prime_5p_at_0

# Function to compute the numerical derivative using the specified formula
def numerical_derivative(formula, f, x_val, order, h=1/128):
    if order == 0:
        return f.subs(x, x_val).evalf()
    elif formula == '2p':
        return (numerical_derivative(formula, f, x_val + h, order - 1) - numerical_derivative(formula, f, x_val - h, order - 1)) / (2 * h)
    elif formula == '3p':
        return (-3 * numerical_derivative(formula, f, x_val, order - 1) + 4 * numerical_derivative(formula, f, x_val + h, order - 1) - numerical_derivative(formula, f, x_val + 2 * h, order - 1)) / (2 * h)
    elif formula == '5p':
        return (-25 * numerical_derivative(formula, f, x_val, order - 1) + 48 * numerical_derivative(formula, f, x_val + h, order - 1) - 36 * numerical_derivative(formula, f, x_val + 2 * h, order - 1) + 16 * numerical_derivative(formula, f, x_val + 3 * h, order - 1) - 3 * numerical_derivative(formula, f, x_val + 4 * h, order - 1)) / (12 * h)

# Compute the first five derivatives at x = 0 using different formulas
derivatives_2p = [numerical_derivative('2p', f, 0, i) for i in range(1, 6)]
derivatives_3p = [numerical_derivative('3p', f, 0, i) for i in range(1, 6)]
derivatives_5p = [numerical_derivative('5p', f, 0, i) for i in range(1, 6)]

print(derivatives_2p, derivatives_3p, derivatives_5p)

