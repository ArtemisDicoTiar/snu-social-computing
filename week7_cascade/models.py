import os

import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go

from week7_cascade.hyperparams import initE, initI, initR, initN, DAYS, BETA, SIGMA, GAMMA, MU


class EpidemicModel:
    def __init__(self, days, initial_conditions, params,  *args, **kwargs):
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

    def _equation(self, *args, **kwargs):
        raise NotImplementedError

    def solve(self, *args, **kwargs):
        raise NotImplementedError

    def plot(self):
        tspan = np.arange(0, self.days, 1)
        sol = self.solve(t=tspan)

        # Create traces
        fig = go.Figure()
        for j, box in zip(range(sol.shape[-1]), self.boxes):
            y_plot = sol[:, j]
            names = {
                "S": "Susceptible",
                "E": "Exposed",
                "I": "Infected",
                "R": "Recovered",
                "D": "Death",
            }
            fig.add_trace(go.Scatter(x=tspan, y=y_plot, mode='lines+markers', name=names[box]))

        if self.days <= 30:
            step = 1
        elif self.days <= 90:
            step = 7
        else:
            step = 30

        # Edit the layout
        fig.update_layout(title=f'Simulation of {self.__class__.__name__} Model',
                          xaxis_title='Day',
                          yaxis_title='Counts',
                          title_x=0.5,
                          width=900, height=600
                          )
        fig.update_xaxes(tickangle=-90, tickformat=None, tickmode='array', tickvals=np.arange(0, self.days + 1, step))
        if not os.path.exists("images"):
            os.mkdir("images")
        fig.write_image(f"images/{self.__class__.__name__.lower()}_simulation.png")
        fig.show()


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
        return res


if __name__ == '__main__':
    seir = SEIR(days=DAYS,
                initial_conditions=["initE", "initI", "initR", "initN"],
                params=["beta", "sigma", "gamma"],
                initE=initE,
                initI=initI,
                initR=initR,
                initN=initN,
                beta=BETA,
                sigma=SIGMA,
                gamma=GAMMA,
                mu=MU)
    seir.plot()
