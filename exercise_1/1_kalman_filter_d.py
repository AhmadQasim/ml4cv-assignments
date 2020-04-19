import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def normal_pdf(x, mu, var):
    return np.exp(-0.5 * (x-mu)**2 / var) /  np.sqrt(2*np.pi*var)


def update(mu1, var1, mu2, var2):
    new_mean = (var2*mu1 + var1*mu2) / (var1 + var2)
    new_var = 1 / (1/var1 + 1/var2)
    return new_mean, new_var


def prediction(mu1, var1, mu2, var2):
    new_mean = mu1 + mu2
    new_var = var1 + var2
    return new_mean, new_var


def main(args):
    measurements = args.sensor_measurements
    x_mu, x_var = 0, 0
    var_sensor, var_action = args.sigma_sensor ** 2, args.sigma_action ** 2

    x_space = np.linspace(0, args.x_space, 200)
    belief = np.zeros_like(x_space)

    for t in range(1, args.timesteps + 1):
        [x_mu, x_var] = prediction(x_mu, x_var, args.velocity, var_action)

        z = measurements[t-1]
        [x_mu, x_var] = update(x_mu, x_var, z, var_sensor)
        belief = np.array([normal_pdf(x, x_mu, x_var) for x in x_space])

        eta = 1 / sum(belief)
        belief = belief * eta

    plt.plot(belief)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--sensor_measurements", default="1.2,1.6,2.5", type=str)
    parser.add_argument("--sigma_sensor", default=0.3, type=int)
    parser.add_argument("--sigma_action", default=0.1, type=int)
    parser.add_argument("--timesteps", default=3, type=int)
    parser.add_argument("--x_space", default=5, type=int)
    parser.add_argument("--velocity", default=1, type=int)

    args = parser.parse_args()

    args.sensor_measurements = [float(x) for x in args.sensor_measurements.split(',')]
    assert len(args.sensor_measurements) == args.timesteps, "sensor measurements should have same length as timesteps"

    main(args)
