import json
import os
from typing import List

import numpy as np

from scipy.integrate import odeint
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_squared_error

from week7_cascade.dataloader import korea_df, korea_pop, us_pop, global_pop, uk_pop, global_df, uk_df, us_df
from week7_cascade.hyperparams import initE, initI, initR, initN, DAYS, BETA, SIGMA, GAMMA, MU


class EpidemicModel:
    def __init__(self, days, initial_conditions, params, *args, **kwargs):
        self.days = days
        self.initial_conditions_names = initial_conditions
        self.params_names = params
        self.initial_conditions = []
        self.params = []

        self.boxes = list(self.__class__.__name__)

        self.__dict__.update(kwargs)
        for initial_condition in initial_conditions:
            self.initial_conditions.append(self.__dict__[initial_condition])

        for param in params:
            self.params.append(self.__dict__[param])

        self.infected_data = None
        self.real_infected_data = None

        self._fig_init()

    def _equation(self, *args, **kwargs):
        raise NotImplementedError

    def solve(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _rmse(real, pred):
        return mean_squared_error(real, pred, squared=False)

    @staticmethod
    def _r2(real, pred):
        return r2_score(real, pred)

    def loss(self):
        real = self.real_infected_data
        pred = self.infected_data
        return {
            "rmse": self._rmse(real, pred),
            "r2": self._r2(real, pred),
        }

    def _fig_init(self):
        self.fig = go.Figure()

    def plot_real(self, data, name):
        tspan = np.arange(0, self.days, 1)
        data = data[:len(tspan)]
        self.real_infected_data = data

        self.fig.add_trace(go.Scatter(x=tspan, y=data, mode='lines+markers', name=name))

    def plot(self,
             title: str = None,
             which: List[str] = None,
             ):
        tspan = np.arange(0, self.days, 1)
        sol = self.solve(t=tspan)

        # Create traces
        for j, box in zip(range(sol.shape[-1]), self.boxes):
            if which is not None and box.lower() not in which:
                continue

            y_plot = sol[:, j]
            names = {
                "S": "Susceptible",
                "E": "Exposed",
                "I": "Infected",
                "R": "Recovered",
                "D": "Death",
            }
            self.fig.add_trace(go.Scatter(x=tspan, y=y_plot, mode='lines+markers', name=names[box]))

        if self.days <= 30:
            step = 1
        elif self.days <= 90:
            step = 7
        else:
            step = 30

        # Edit the layout
        self.fig.update_layout(title=f'{self.__class__.__name__} Model {f"({title})" if title else ""}',
                               xaxis_title='Day',
                               yaxis_title='Counts',
                               title_x=0.5,
                               width=900, height=600
                               )
        self.fig.update_xaxes(tickangle=-90, tickformat=None, tickmode='array',
                              tickvals=np.arange(0, self.days + 1, step))

    def save_plot(self, save_fname: str, show: bool = False):
        if not os.path.exists("images"):
            os.mkdir("images")
        if not os.path.exists(f"images/{self.__class__.__name__.lower()}"):
            os.mkdir(f"images/{self.__class__.__name__.lower()}")
        self.fig.write_image(
            f"images/{self.__class__.__name__.lower()}/{save_fname if save_fname else 'simulation'}.png")
        if show:
            self.fig.show()


class SIR(EpidemicModel):
    def __init__(self, days, initial_conditions, params, *args, **kwargs):
        super().__init__(days, initial_conditions, params, *args, **kwargs)

    def _equation(self, *args, **kwargs):
        # z, t, beta, gamma = kwargs["z"], kwargs["t"], kwargs["beta"], kwargs["gamma"]
        z, t, beta, gamma = args
        S, I, R = z
        N = S + I + R
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    def solve(self, *args, **kwargs):
        t, initial_conditions, params = kwargs["t"], self.initial_conditions, self.params
        initI, initR, initN = initial_conditions
        beta, gamma = params
        initS = initN - (initI + initR)
        res = odeint(self._equation, [initS, initI, initR], t, args=(beta, gamma))
        self.infected_data = res[:, 1]
        return res


class SEIR(EpidemicModel):
    def __init__(self, days, initial_conditions, params, *args, **kwargs):
        super().__init__(days, initial_conditions, params, *args, **kwargs)

    def _equation(self, *args, **kwargs):
        # z, t, beta, sigma, gamma = kwargs["z"], kwargs["t"], kwargs["beta"], kwargs["sigma"], kwargs["gamma"]
        z, t, beta, sigma, gamma = args
        S, E, I, R = z
        N = S + E + I + R
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]

    def solve(self, *args, **kwargs):
        t, initial_conditions, params = kwargs["t"], self.initial_conditions, self.params
        initE, initI, initR, initN = initial_conditions
        beta, sigma, gamma = params
        initS = initN - (initE + initI + initR)
        res = odeint(self._equation, [initS, initE, initI, initR], t, args=(beta, sigma, gamma))
        self.infected_data = res[:, 2]
        return res


class SEIRD(EpidemicModel):
    def __init__(self, days, initial_conditions, params, *args, **kwargs):
        super().__init__(days, initial_conditions, params, *args, **kwargs)

    def _equation(self, *args, **kwargs):
        # z, t, beta, sigma, gamma, mu = kwargs["z"], kwargs["t"], kwargs["beta"], kwargs["sigma"], kwargs["gamma"], \
        #                                kwargs["mu"]
        z, t, beta, sigma, gamma, mu = args
        S, E, I, R, D = z
        N = S + E + I + R + D
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I - mu * I
        dRdt = gamma * I
        dDdt = mu * I
        return [dSdt, dEdt, dIdt, dRdt, dDdt]

    def solve(self, *args, **kwargs):
        t, initial_conditions, params = kwargs["t"], self.initial_conditions, self.params
        initE, initI, initR, initN, initD = initial_conditions
        beta, sigma, gamma, mu = params
        initS = initN - (initE + initI + initR + initD)
        res = odeint(self._equation, [initS, initE, initI, initR, initD], t, args=(beta, sigma, gamma, mu))
        self.infected_data = res[:, 2]
        return res


if __name__ == '__main__':
    country = "korea"
    real_range = [0, 150]
    population = {
        "korea": korea_pop,
        "us": us_pop,
        "uk": uk_pop,
        "global": global_pop
    }[country]
    country_df = {
        "korea": korea_df,
        "us": us_df,
        "uk": uk_df,
        "global": global_df
    }[country]
    N_range = np.arange(0.1, 1, 0.25)
    E_range = np.arange(0, 50, 50)
    I_range = np.arange(0, 50, 50)
    beta_range = np.arange(0.01, 1, 0.03)
    sigma_range = np.arange(0.01, 1, 0.03)
    gamma_range = np.arange(0.01, 1, 0.03)
    mu_range = np.arange(0.01, 0.03, 0.01)

    losses = {}
    model = SEIR(days=max(real_range),
                 initial_conditions=["initE", "initI", "initR", "initN"],
                 params=["beta", "sigma", "gamma"],
                 # infection rate, incubation rate, recovery rate
                 initE=initE,
                 initI=initI,
                 initR=initR,
                 initN=population,
                 beta=BETA,
                 sigma=SIGMA,
                 gamma=GAMMA,
                 mu=MU)

    if not os.path.exists(f"losses"):
        os.mkdir(f"losses")
    if not os.path.exists(f"losses/{model.__class__.__name__.lower()}"):
        os.mkdir(f"losses/{model.__class__.__name__.lower()}")
    if not os.path.exists(f"losses/{model.__class__.__name__.lower()}/{country}"):
        os.mkdir(f"losses/{model.__class__.__name__.lower()}/{country}")

    print(f"total search: {len(beta_range) * len(sigma_range) * len(gamma_range) * len([0]) * len(E_range) * len(I_range) * len(N_range)}")
    for beta in beta_range:
        for sigma in sigma_range:
            for gamma in gamma_range:
                # for mu in mu_range:
                mu=0
                for E in E_range:
                    for I in I_range:
                        for N in N_range:
                            # beta = r0 * gamma
                            model = SEIR(days=max(real_range),
                                         initial_conditions=["initE", "initI", "initR", "initN"],
                                         params=["beta", "sigma", "gamma"],
                                         initE=initE + E,
                                         initI=initI + I,
                                         initR=initR,
                                         initN=population*N,
                                         beta=beta,
                                         sigma=sigma,
                                         gamma=gamma,
                                         mu=mu
                                         )
                            real_data = (country_df["Confirmed"] - (country_df["Recovered"] + country_df["Deaths"])).values
                            real_data = real_data[real_range[0]: real_range[1]]
                            base = real_data[real_range[0]]
                            real_data -= base
                            model.plot_real(real_data, "real")
                            model.plot()
                            loss = model.loss()
                            losses[f"E{E}_I{I}_N{N}_b{round(beta, 3)}_s{round(sigma, 3)}_g{round(gamma, 3)}_m{round(mu, 3)}"] = loss

                            if not os.path.exists(f"images/{model.__class__.__name__.lower()}/{country}"):
                                os.mkdir(f"images/{model.__class__.__name__.lower()}/{country}")
                            # model.save_plot(f"{country}/E{E}_I{I}_b{round(beta, 3)}_s{round(sigma, 3)}_g{round(gamma, 3)}_m{round(mu, 3)}", show=True)
                            print(".", end="")
                        print()
                    print()
                    # print()
                print()
            print()
        print()
    losses = sorted(losses.items(), key=lambda i: i[1]["r2"], reverse=True)
    with open(f"losses/{model.__class__.__name__.lower()}/{country}/{real_range}.json", "w") as f:
        f.write(json.dumps(losses))

    opt = losses[0]
    E, I, N, beta, sigma, gamma, mu = opt[0].split("_")
    E, I, N, beta, sigma, gamma, mu = float(E[1:]), float(I[1:]), float(N[1:]), float(beta[1:]), float(sigma[1:]), float(gamma[1:]), float(mu[1:])
    # E, I, beta, sigma, gamma, mu = 0, 0, 0.86, 0.19, 0.34, 0
    r0 = beta / gamma
    model = SEIR(days=max(real_range),
                 initial_conditions=["initE", "initI", "initR", "initN"],
                 params=["beta", "sigma", "gamma"],
                 initE=initE + E,
                 initI=initI + I,
                 initR=initR,
                 initN=population * N,
                 beta=beta,
                 sigma=sigma,
                 gamma=gamma,
                 mu=mu)
    real_data = (country_df["Confirmed"] - (country_df["Recovered"] + country_df["Deaths"])).values
    real_data = real_data[real_range[0]: real_range[1]]
    base = real_data[real_range[0]]
    real_data -= base
    model.plot_real(real_data, "real")
    model.plot(which=["i"], title=country)

    if not os.path.exists(f"images/{model.__class__.__name__.lower()}/{country}"):
        os.mkdir(f"images/{model.__class__.__name__.lower()}/{country}")
    model.save_plot(f"{country}/{real_range}r0{round(r0, 3)}E{E}I{I}N{N}b{round(beta, 3)}s{round(sigma, 3)}g{round(gamma, 3)}m{round(mu, 3)}",
                    show=True)