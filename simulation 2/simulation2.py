import numpy as np
from scipy.stats import expon, weibull_min, gamma
from scipy.special import gamma as gamma_fun

def generate_parameter_sets_weibull(m, base_lambda, base_mu, base_k):
    """
    Generates m sets of parameters such that the expected values of X1 and X2
    are increasingly greater than the expected values of Y, Y1, and Y2.
    Using Weibull distributions for Y, Y1, and Y2.

    Args:
        m: The number of parameter sets to generate.
        base_lambda: Base value for the lambda parameters of X1 and X2.
        base_mu: Base value for the mu (scale) parameters of Y, Y1, and Y2.
        base_k: Base value for the k (shape) parameters of Y, Y1, and Y2.

    Returns:
        A list of tuples, where each tuple contains a set of parameters:
        (lambda1, lambda2, mu, k, mu1, k1, mu2, k2)
    """

    parameter_sets = []
    for i in range(m):
        # Increase lambda for X1 and X2 more significantly
        lambda1 = base_lambda / (1 + i * 0.5)  # Faster increase for X1
        lambda2 = base_lambda / (1 + i * 0.3)  # Slower increase for X2

        # Slightly decrease mu and k for Y, Y1, and Y2
        mu = base_mu / (1 + i * 0.5)  # Slower decrease
        k = base_k * (1 + i * 0.3)  # Even slower decrease
        mu1 = base_mu / (1 + i * 0.4)
        k1 = base_k * (1 + i * 0.6)

        mu2 = base_mu / (1 + i * 0.2)
        k2 = base_k * (1 + i * 0.8)
        parameter_sets.append((lambda1, lambda2, mu, k, mu1, k1, mu2, k2))

    return parameter_sets


def generate_parameter_sets_gamma(m, base_lambda, base_alpha, base_theta):
    """
    Generates m sets of parameters such that the expected values of X1 and X2
    are increasingly greater than the expected values of Y, Y1, and Y2.

    Args:
        m: The number of parameter sets to generate.
        base_lambda: Base value for the lambda parameters of X1 and X2.
        base_alpha: Base value for the alpha parameters of Y, Y1, and Y2.
        base_theta: Base value for the theta parameters of Y, Y1, and Y2.

    Returns:
        A list of tuples, where each tuple contains a set of parameters:
        (lambda1, lambda2, alpha, theta, alpha1, theta1, alpha2, theta2)
    """

    parameter_sets = []
    for i in range(m):
        # Increase lambda for X1 and X2 more significantly
        lambda1 = base_lambda / (1 + i * 0.5)  # Faster increase for X1
        lambda2 = base_lambda / (1 + i * 0.4)  # Slower increase for X2

        # Slightly decrease alpha and theta for Y, Y1, and Y2
        alpha = base_alpha / (1 + i * 0.05)  # Slower decrease
        theta = base_theta / (1 + i * 0.03)  # Even slower decrease
        alpha1 = base_alpha / (1 + i * 0.04)
        theta1 = base_theta / (1 + i * 0.06)

        alpha2 = base_alpha / (1 + i * 0.02)
        theta2 = base_theta / (1 + i * 0.08)
        parameter_sets.append((lambda1, lambda2, alpha, theta, alpha1, theta1, alpha2, theta2))

    return parameter_sets

def generate_parameter_sets_expon(m, base_lambda, base_theta):

    parameter_sets = []
    for i in range(m):
        # Increase lambda for X1 and X2 more significantly
        lambda1 = base_lambda / (1 + i * 0.5)  # Faster increase for X1
        lambda2 = lambda1  # Slower increase for X2

        # Slightly decrease alpha and theta for Y, Y1, and Y2
        theta = base_theta * (1 + i * 0.06)  
        theta1 = theta
        theta2 = theta

        parameter_sets.append((lambda1, lambda2, theta, theta1, theta2))

    return parameter_sets

def generate_parameter_sets_const(m, base_lambda, base_c):

    parameter_sets = []
    for i in range(m):
        # Increase lambda for X1 and X2 more significantly
        lambda1 = base_lambda / (1 + i * 0.3)  # Faster increase for X1
        lambda2 = lambda1  # Slower increase for X2

        # Slightly decrease alpha and theta for Y, Y1, and Y2
        c = base_c / (1 + i * 0.15 )  
        c1 = c
        c2 = c

        parameter_sets.append((lambda1, lambda2, c, c1, c2))

    return parameter_sets


def asymp_mean_val(lambda1, lambda2, Y, Y1, Y2):
    return 1 / (lambda1 * lambda2 * (Y1.mean() + Y2.mean() + Y.var()/ Y.mean() + Y.mean()))



fact = [1]
for i in range(1,21):
    fact.append(i * fact[-1])

def LT_weibull(lamb, mu, k):
    p = 0
    for i in range(20):
        p += ((- lamb * mu) ** i) / fact[i] * gamma_fun(1 + i / k)
    return p

def P_tau_leq_tauc_weibull(lambda1, lambda2, mu, k, mu1, k1, mu2, k2):

    p1 = LT_weibull(lambda1,mu2,k2) * (LT_weibull(lambda1,mu,k) - LT_weibull(lambda1 + lambda2,mu,k))

    p2 = LT_weibull(lambda2,mu1,k1) * (LT_weibull(lambda2,mu,k) - LT_weibull(lambda1 + lambda2,mu,k))

    return 1 - LT_weibull(lambda1 + lambda2, mu, k) - (p1 + p2)



def LT_gamma(lamb, alpha, theta):
    
    return 1 / (1 + theta * lamb) ** alpha


def P_tau_leq_tauc_gamma(lambda1, lambda2, alpha, theta, alpha1, theta1, alpha2, theta2):
    
    p1 = LT_gamma(lambda1,alpha2,theta2) * (LT_gamma(lambda1,alpha,theta) - LT_gamma(lambda1 + lambda2,alpha,theta))

    p2 = LT_gamma(lambda2,alpha1,theta1) * (LT_gamma(lambda2,alpha,theta) - LT_gamma(lambda1 + lambda2,alpha,theta))

    return 1 - LT_gamma(lambda1 + lambda2, alpha, theta) - (p1 + p2)



def LT_expon(lamb, theta):
    
    return theta / (theta + lamb)


def P_tau_leq_tauc_expon(lambda1, lambda2, theta, theta1, theta2):
    
    p1 = LT_expon(lambda1,theta2) * (LT_expon(lambda1,theta) - LT_expon(lambda1 + lambda2,theta))

    p2 = LT_expon(lambda2,theta1) * (LT_expon(lambda2,theta) - LT_expon(lambda1 + lambda2,theta))

    return 1 - LT_expon(lambda1 + lambda2,  theta) - (p1 + p2)



class const():

    def __init__(self, c):
        self.c = c
        return

    def rvs(self):
        return self.c
    
    def mean(self):
        return self.c
    
    def var(self):
        return 0
    

def LT_const(lamb, c):
    
    return np.e ** (- lamb * c)


def P_tau_leq_tauc_const(lambda1, lambda2, c, c1, c2):
    
    p1 = LT_const(lambda1,c2) * (LT_const(lambda1,c) - LT_const(lambda1 + lambda2,c))

    p2 = LT_const(lambda2,c1) * (LT_const(lambda2,c) - LT_const(lambda1 + lambda2,c))

    return 1 - LT_const(lambda1 + lambda2,  c) - (p1 + p2)

    



if __name__ == "__main__":

    m = 30  # Number of experiments
    n = 100000  # Number of samples

    base_lambda = 1
    base_mu = 1.128
    base_k = 2
    base_alpha = 1
    base_theta = 1 
    base_c = 1

    repair_distribution = "constant"

    if repair_distribution == "weibull":
        params = generate_parameter_sets_weibull(m, base_lambda, base_mu, base_k)
    elif repair_distribution == "exponential":
        params = generate_parameter_sets_expon(m, base_lambda, base_theta)
    elif repair_distribution == "constant":
        params = generate_parameter_sets_const(m, base_lambda, base_c)
    else: # Gamma
        params = generate_parameter_sets_gamma(m, base_lambda, base_alpha, base_theta)

    for i in range(m):

        if repair_distribution == "weibull":
            lambda1, lambda2, mu, k, mu1, k1, mu2, k2 = params[i]
        elif repair_distribution == "exponential":
            lambda1, lambda2, theta, theta1, theta2 = params[i]
        elif repair_distribution == "constant":
            lambda1, lambda2, c, c1, c2 = params[i]
        else: # Gamma
            lambda1, lambda2, alpha, theta, alpha1, theta1, alpha2, theta2 = params[i]


        times = 0 # Guardar la suma de los ciclos de regeneracion

        # Definir las variables aleatorias
        X1 = expon(scale =1/lambda1)
        X2 = expon(scale =1/lambda2)

        if repair_distribution == "weibull":
            Y = weibull_min(c = k, scale = mu)
            Y1 = weibull_min(c = k1, scale = mu1)
            Y2 = weibull_min(c = k2, scale = mu2)
        elif repair_distribution == "exponential":
            Y = expon(scale =1/theta)
            Y1 = expon(scale =1/theta)
            Y2 = expon(scale =1/theta)
        elif repair_distribution == "constant":
            Y = const(c)
            Y1 = const(c1)
            Y2 = const(c2)
        else: # Gamma
            Y = gamma(a = alpha, scale = theta)
            Y1 = gamma(a = alpha1, scale = theta1)
            Y2 = gamma(a = alpha2, scale = theta2)

        # Realizar los n ciclos de regeneracion
        for _ in range(n):

            # En el instante inicial comienza una reparacion Tipo 2 y gracias a la
            # ausencia de memoria las exponenciales distribuyen igual
            x1 = X1.rvs()
            x2 = X2.rvs()
            y = Y.rvs()

            # Si ambas componentes fallan luego de la reparacion de Tipo 2
            if min(x1, x2) > y:
                times += y

            # Si la componente 1 falla antes de la reparacion de Tipo 2
            elif x1 < y:
                y1 = Y1.rvs()

                # Si la componente 2 falla luego de la reparacion de Tipo 2 
                # y luego de la reparacion de la componente que fallo
                if x2 > y + y1:
                    times += y + y1

                else:
                    times += x2

            # Si la componente 2 falla antes de la reparacion de Tipo 2
            elif x2 < y:
                y2 = Y2.rvs()

                # Si la componente 1 falla luego de la reparacion de Tipo 2 
                # y luego de la reparacion de la componente que fallo
                if x1 > y + y2:
                    times += y + y2

                else:
                    times += x1

            # Caso imposible
            else :
                print('what???')



        At = asymp_mean_val(lambda1, lambda2, Y, Y1, Y2)

        if(repair_distribution == "weibull"):
            Et = times / n / P_tau_leq_tauc_weibull(lambda1, lambda2, mu, k, mu1, k1, mu2, k2)
        elif repair_distribution == "exponential":
            Et = times / n / P_tau_leq_tauc_expon(lambda1, lambda2, theta, theta1, theta2)
        elif repair_distribution == "constant":
            Et = times / n / P_tau_leq_tauc_const(lambda1, lambda2, c, c1, c2)
        else:
            Et = times / n / P_tau_leq_tauc_gamma(lambda1, lambda2, alpha, theta, alpha1, theta1, alpha2, theta2)

        r = min(X1.mean(), X2.mean()) / max((Y.mean(), Y1.mean(), Y2.mean()))
        print(i, r, n, times, Et, At, abs(Et - At)/Et)
