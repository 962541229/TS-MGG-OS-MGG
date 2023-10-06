import numpy as np

# import sympy as sm
# from sympy.abc import a, x


class InformationLoss:
    def __init__(self, alpha, beta, m, domain):
        super(InformationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.m = m
        self.domain = domain

    # def guassian(self, theta, c):
    #     """returns information loss for Guassian function."""
    #     # func = MembershipFuncs(x)
    #     # # sm.exp((- (x - c) ** 2) / (2 * theta ** 2))
    #     # guassian_func = func.gaussian_func(theta, c)
    #     # f_alpha = guassian_func - self.alpha
    #     # f_beta = guassian_func - self.beta
    #     # f_m = guassian_func - self.m
    #     # sol_alpha = sm.solve(f_alpha, x)
    #     # sol_beta = sm.solve(f_beta, x)
    #     # sol_m = sm.solve(f_m, x)
    #     # print(sol_alpha)
    #     # print(sol_beta)
    #     # print(sol_m)
    #     # elevated_loss = sm.integrate(1 - guassian_func, (x, sol_alpha[0], sol_alpha[1])) + \
    #     #                 sm.integrate(m - guassian_func, (x, sol_beta[0], sol_m[0])) + \
    #     #                 sm.integrate(m - guassian_func, (x, sol_m[1], sol_beta[1]))
    #     # reduced_loss = sm.integrate(guassian_func, (x, self.domain[0], sol_beta[0])) + \
    #     #                sm.integrate(m - guassian_func, (x, sol_beta[1], self.domain[1])) + \
    #     #                sm.integrate(guassian_func - m, (x, sol_m[0], sol_alpha[0])) + \
    #     #                sm.integrate(guassian_func - m, (x, sol_alpha[1], sol_m[1]))
    #     # print(elevated_loss.evalf())
    #     # print(reduced_loss.evalf())


if __name__ == '__main__':
    alpha_1 = 0.75
    beta_1 = 0.25
    m_1 = 0.5
    domain_1 = [0, 10]
    theta = 2
    c = 5
    info_loss = InformationLoss(alpha_1, beta_1, m_1, domain_1)
    # info_loss.guassian(theta, c)