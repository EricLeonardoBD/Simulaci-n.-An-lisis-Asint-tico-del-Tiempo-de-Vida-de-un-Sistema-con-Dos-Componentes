import numpy as np
from scipy.stats import expon, weibull_min, gamma, expon, kstest, anderson, ecdf
import matplotlib.pyplot as plt
#import statsmodels.api as sm

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


def generate_parameter_sets_expon(m, base_lambda):
    """
    Generates m sets of parameters for exponential distributions such that the
    expected values of X1 and X2 are increasingly greater than the expected
    values of Y, Y1, and Y2.  For exponential distributions, only the rate
    parameter (lambda) controls the expected value (mean = 1/lambda).

    We'll simulate the other distributions (Y, Y1, Y2) using exponentials as well,
    for simplicity and to maintain the focus on varying the exponential rates.

    Args:
        m: The number of parameter sets to generate.
        base_lambda: Base value for the lambda parameters of Y, Y1, and Y2.
                      X1 and X2 will have *decreasing* lambda values, meaning *increasing* means.

    Returns:
        A list of tuples, where each tuple contains a set of parameters:
        (lambda1, lambda2, lambdaY, lambdaY1, lambdaY2)
    """

    parameter_sets = []
    for i in range(m):
        # Decrease lambda for X1 and X2 to increase their expected values significantly
        lambda1 = base_lambda / (1 + i * 0.5)  # Faster increase for mean(X1)
        lambda2 = base_lambda / (1 + i * 0.4)  # Slower increase for mean(X2)

        # Slightly increase lambda for Y, Y1, and Y2 to *decrease* their expected values
        lambdaY = base_lambda * (1 + i * 0.5)  # Slower decrease for mean(Y)
        lambdaY1 = base_lambda * (1 + i * 0.4) # a little faster
        lambdaY2 = base_lambda * (1 + i * 0.3) # even faster

        parameter_sets.append((lambda1, lambda2, lambdaY, lambdaY1, lambdaY2))

    return parameter_sets


def asymp_mean_val(lambda1, lambda2, Y, Y1, Y2):
    return 1 / (lambda1 * lambda2 * (Y1.mean() + Y2.mean() + Y.var()/ Y.mean() + Y.mean()))




if __name__ == "__main__":

    m = 30 # Number of experiments
    n = 300000  # Number of samples

    base_lambda = 1
    base_mu = 1.128
    base_k = 2
    base_alpha = 1
    base_theta = 1 

    repair_distribution = "exponential"

    if repair_distribution == "weibull":
        params = generate_parameter_sets_weibull(m, base_lambda, base_mu, base_k)
    elif repair_distribution == "exponential":
        params = generate_parameter_sets_expon(m,base_lambda)
    else: # Gamma
        params = generate_parameter_sets_gamma(m, base_lambda, base_alpha, base_theta)

    print("Reparaciones con distribucion " + repair_distribution)

    for i in range(0 ,m):
        print(" ---- + ---- + ---- + ---- + ---- + ---- + ---- + ---- + ---- + ---- ")

        if repair_distribution == "weibull":
            lambda1, lambda2, mu, k, mu1, k1, mu2, k2 = params[i]
        elif repair_distribution == "exponential":
            lambda1, lambda2, mu, mu1, mu2 = params[i] 
        else: # Gamma
            lambda1, lambda2, alpha, theta, alpha1, theta1, alpha2, theta2 = params[i]


        times = 0 
        stimes = 0
        tau = [0]  # taus
        # Definir las variables aleatorias
        X1 = expon(scale =1/lambda1)
        X2 = expon(scale =1/lambda2)

        if repair_distribution == "weibull":
            Y = weibull_min(c = k, scale = mu)
            Y1 = weibull_min(c = k1, scale = mu1)
            Y2 = weibull_min(c = k2, scale = mu2)
        elif repair_distribution == "exponential":
            Y = expon(scale = 1/mu)
            Y1 = expon(scale = 1/mu1)
            Y2 = expon(scale = 1/mu2)
        else: # Gamma
            Y = gamma(a = alpha, scale = theta)
            Y1 = gamma(a = alpha1, scale = theta1)
            Y2 = gamma(a = alpha2, scale = theta2)

        # Realizar los n ciclos de regeneracion
        while times < n or len(tau) == 1:

            x1 = X1.rvs()
            x2 = X2.rvs()

            s = 0
            # Realizar todos los ciclos de regeneracion con reparaciones de Tipo 2
            # antes de que falle una componente
            while s < min(x1, x2) and (times < n or len(tau) == 1):

                # En caso de no ser la primera iteracion guardar la duracion del ultimo ciclo
                if s != 0:
                    times += 1
                    stimes += y

                y = Y.rvs()
                s += y


            if times >= n:
                break

            # Ambas componentes fallaron antes de la ultima reparacion de Tipo 2
            elif s > max(x1, x2):
                times += 1
                stimes += y
                tau.append(stimes)
                stimes = 0
            
            # Solo fallo la primera componente
            elif s > x1:
                y1 = Y1.rvs()
                s += y1

                # No dio tiempo a reparar la primera componente
                if s > x2:
                    times += 1
                    stimes += x2 - (s-y1)
                    tau.append(stimes)
                    stimes = 0

                # Si dio tiempo a reparar la primera componente
                else:
                    times += 1
                    stimes += y1

            # Solo fallo la segunda componente
            else:
                y2 = Y2.rvs()
                s += y2

                # No dio tiempo a reparar la segunda componente
                if s > x1:
                    times += 1
                    stimes += x1 - (s-y2)
                    tau.append(stimes)
                    stimes = 0

                # Si dio tiempo a reparar la segunda componente
                else:
                    times += 1
                    stimes += y2

        At = asymp_mean_val(lambda1, lambda2, Y, Y1, Y2)
        r = min(X1.mean(), X2.mean()) / max((Y.mean(), Y1.mean(), Y2.mean()))
        #Et = sum(times)/fails 
        print(i, r, len(tau))

        # 1. Datos de ejemplo (reemplázalos por tus datos reales)
        datos = np.array(tau[1:])

        # 2. Parámetros TEÓRICOS de tu exponencial (λ = tasa, scale = 1/λ)
        lambda_teorico = 1 / At 
        scale_teorico =  At 
        loc_teorico = 0 

        # 3. Crear distribución exponencial teórica
        dist_teorica = expon(loc=loc_teorico, scale=scale_teorico)

        # --- Prueba de Kolmogorov-Smirnov (KS) ---
        ks_stat, p_value_ks = kstest(datos, dist_teorica.cdf)
        print("\n--- Prueba de Kolmogorov-Smirnov (KS) ---")
        print(f"Estadístico KS: {ks_stat:.4f}")
        print(f"p-valor: {p_value_ks:.4f}")
        if p_value_ks > 0.05:
            print("✅ No se rechaza H0: Los datos podrían seguir la exponencial teórica (p > 0.05)")
        else:
            print("❌ Se rechaza H0: Los datos NO siguen la exponencial teórica (p ≤ 0.05)")

        # --- Prueba de Anderson-Darling (AD) ---
        anderson_stat = anderson(datos, dist='expon')
        print("\n--- Prueba de Anderson-Darling (AD) ---")
        print(f"Estadístico AD: {anderson_stat.statistic:.4f}")
        print("Valores críticos:", anderson_stat.critical_values)
        print("Niveles de significancia (%):", anderson_stat.significance_level)

        # Comparar con el valor crítico para alpha=0.05 (índice 2 en la lista)
        if anderson_stat.statistic > anderson_stat.critical_values[2]:
            print("❌ Se rechaza H0: Los datos NO siguen la exponencial teórica (alpha=0.05)")
        else:
            print("✅ No se rechaza H0: Los datos podrían seguir la exponencial teórica (alpha=0.05)")

        """
        # --- Gráficos de diagnóstico ---
        plt.figure(figsize=(12, 4))

        # Histograma + PDF teórica
        plt.subplot(1, 2, 1)
        #plt.hist(datos, bins='auto', density=True, alpha=0.6, color='blue', label='Datos')
        x = np.linspace(0, max(datos), 1000)
        plt.plot(x, ecdf(datos).cdf.evaluate(x), color='blue', label=f"Datos (r = {r}, fallos = {len(tau)})")
        plt.plot(x, dist_teorica.cdf(x), 'r-', lw=2, label=f'Exponencial (λ={lambda_teorico})')
        plt.title('Ajuste: Datos vs. Exponencial Teórica')
        plt.legend()

        # Gráfico Q-Q
        #plt.subplot(1, 2, 2)
        #sm.qqplot(datos, dist=dist_teorica, line='45', fit=True, marker='o', alpha=0.5)
        #plt.title('Gráfico Q-Q')

        plt.tight_layout()
        plt.show()
        """