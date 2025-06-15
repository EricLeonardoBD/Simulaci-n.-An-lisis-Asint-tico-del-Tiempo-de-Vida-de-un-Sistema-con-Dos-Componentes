import numpy as np
from scipy.stats import expon, weibull_min, gamma

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


def asymp_mean_val(lambda1, lambda2, Y, Y1, Y2):
    return 1 / (lambda1 * lambda2 * (Y1.mean() + Y2.mean() + Y.var()/ Y.mean() + Y.mean()))


if __name__ == "__main__":

    m = 30  # Number of experiments
    n = 300000  # Number of samples

    base_lambda = 1
    base_mu = 1.128
    base_k = 2
    base_alpha = 1
    base_theta = 1 

    repair_distribution = "gamma"

    if repair_distribution == "weibull":
        params = generate_parameter_sets_weibull(m, base_lambda, base_mu, base_k)
    else: # Gamma
        params = generate_parameter_sets_gamma(m, base_lambda, base_alpha, base_theta)

    for i in range(m):

        if repair_distribution == "weibull":
            lambda1, lambda2, mu, k, mu1, k1, mu2, k2 = params[i]
        else: # Gamma
            lambda1, lambda2, alpha, theta, alpha1, theta1, alpha2, theta2 = params[i]


        times = [] # Guardar la duracion de cada ciclo de regeneracion
        fails = 0  # Numero de fallos ocurridos

        # Definir las variables aleatorias
        X1 = expon(scale =1/lambda1)
        X2 = expon(scale =1/lambda2)

        if repair_distribution == "weibull":
            Y = weibull_min(c = k, scale = mu)
            Y1 = weibull_min(c = k1, scale = mu1)
            Y2 = weibull_min(c = k2, scale = mu2)
        else: # Gamma
            Y = gamma(a = alpha, scale = theta)
            Y1 = gamma(a = alpha1, scale = theta1)
            Y2 = gamma(a = alpha2, scale = theta2)

        # Realizar los n ciclos de regeneracion
        while len(times) < n or fails == 0:

            x1 = X1.rvs()
            x2 = X2.rvs()

            s = 0
            # Realizar todos los ciclos de regeneracion con reparaciones de Tipo 2
            # antes de que falle una componente
            while s < min(x1, x2) and (len(times) < n or fails == 0):

                # En caso de no ser la primera iteracion guardar la duracion del ultimo ciclo
                if s != 0:
                    times.append(y) 

                y = Y.rvs()
                s += y


            if len(times) >= n:
                break

            # Ambas componentes fallaron antes de la ultima reparacion de Tipo 2
            elif s > max(x1, x2):
                fails += 1
                times.append(max(x1,x2) - (s - y))
            
            # Solo fallo la primera componente
            elif s > x1:
                y1 = Y1.rvs()
                s += y1

                # No dio tiempo a reparar la primera componente
                if s > x2:
                    fails += 1
                    times.append(x2 - (s-y1))

                # Si dio tiempo a reparar la primera componente
                else:
                    times.append(y1)

            # Solo fallo la segunda componente
            else:
                y2 = Y2.rvs()
                s += y2

                # No dio tiempo a reparar la segunda componente
                if s > x1:
                    fails += 1
                    times.append(x1 - (s-y2))

                # Si dio tiempo a reparar la segunda componente
                else:
                    times.append(y2)

        At = asymp_mean_val(lambda1, lambda2, Y, Y1, Y2)
        r = min(X1.mean(), X2.mean()) / max((Y.mean(), Y1.mean(), Y2.mean()))
        Et = sum(times)/fails 
        print(i, r, len(times), sum(times), fails, Et, At, abs(Et - At)/Et)
