import numpy as np
import argparse

TWO_PI = np.pi*2


def compute_rotations(p0, p1, p2, p3, x_0, x_1, ansatz):
    if ansatz == '2A':
        # alpha + omega_0 * x_0 + omega_1 * x_1
        R_y = np.mod(p0 + p1 * x_0 + p2 * x_1, TWO_PI)
        # phi
        R_z = np.mod(p3, TWO_PI)
    elif ansatz == '2Ap':
        # amplitude method: population = 0.5 - 0.5 * numpy.exp(-0.09 * A/200) * numpy.cos(6.94 * A/200 - 5.71)
        # for A in 155mV - 255 mV
        # => A_y in 1.55 dV - 2.55 dV
        A_y = np.mod(p0 + p1 * x_0 + p2 * x_1, 0.9) + 1.65
        R_y = np.arcsin(0.5*(1-np.cos(6.94 * A_y / 2 - 5.71)))
        R_z = np.mod(p3, TWO_PI)
    elif ansatz == '2B':
        # alpha + omega_0 * x_0
        R_y = np.mod(p0 + p1 * x_0, TWO_PI)
        # alpha + omega_1 * x_1
        R_z = np.mod(p2 + p3 * x_1, TWO_PI)
    elif ansatz == '2Bp':
        # amplitude method: population = 0.5 - 0.5 * numpy.exp(-0.09 * A/200) * numpy.cos(6.94 * A/200 - 5.71)
        # for A in 155mV - 255 mV
        # => A_y in 1.55 dV - 2.55 dV
        A_y = np.mod(p0 + p1 * x_0, 1.8) + 1.65
        R_y = np.arcsin(0.5*(1-np.cos(6.94 * A_y / 2 - 5.71))) * 2
        if A_y > 2.55:
            R_y += np.pi
        R_z = np.mod(p2 + p3 * x_1, TWO_PI)
    elif ansatz == '2C':
        # omega_0 * x_0 + omega_1 * x_1
        R_y = np.mod(p0 * x_0 + p1 * x_1, TWO_PI)
        # omega_2 * x_0 + omega_3 * x_1
        R_z = np.mod(p2 * x_0 + p3 * x_1, TWO_PI)
    elif ansatz == '2Cp':
        # amplitude method: population = 0.5 - 0.5 * numpy.exp(-0.09 * A/200) * numpy.cos(6.94 * A/200 - 5.71)
        # for A in 155mV - 255 mV
        # => A_y in 1.55 dV - 2.55 dV
        A_y = np.mod(p0 * x_0 + p1 * x_1, 1.8) + 1.65
        R_y = np.arcsin(0.5*(1-np.cos(6.94 * A_y / 2 - 5.71))) * 2
        if A_y > 2.55:
            R_y += np.pi
        R_z = np.mod(p2 * x_0 + p3 * x_1, TWO_PI)
    elif ansatz == '2D':
        # alpha + omega_0 * x_0 + omega_1 * x_1
        R_y = np.mod(p0, TWO_PI)
        # phi
        R_z = np.mod(p1 * x_0 + p2 * x_1 + p3, TWO_PI)
    else:
        raise NotImplemented
    # print(round(R_y/np.pi*180), round(R_z/np.pi*180))
    return R_y, R_z


def generate_angles(data, optimal_parameters, ansatz):
    n_testdata = len(data)
    n_layers = len(optimal_parameters) // 4

    angles = np.zeros((n_testdata, n_layers*2))

    for row, point in enumerate(data):
        x_0, x_1 = point
        for i in range(n_layers):
            p0, p1, p2, p3 = optimal_parameters[i*4:(i+1)*4]
            R_y, R_z = compute_rotations(p0, p1, p2, p3, x_0, x_1, ansatz)
            angles[row, i*2:(i+1)*2] = R_y, R_z

    return angles


def load_data(name, test_or_training="test"):
    data = np.loadtxt(f'{name}-{test_or_training}_data.txt')

    return data


def load_optimal_parameters(name):
    optimal_parameters = np.loadtxt(f'{name}-parameters.txt')

    return optimal_parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help="Name of experiment (e.g. circle-2A-2_layers)")
    parser.add_argument('-o', type=str, required=True, help="Output file")
    parser.add_argument('--ansatz', type=str, default='2A', help="Ansatz to use")
    args = parser.parse_args()

    data = load_data(args.name)
    optimal_parameters = load_optimal_parameters(args.name)
    angles = generate_angles(data, optimal_parameters, args.ansatz)

    np.savetxt(args.o, angles, delimiter=',', fmt='%5.3f')

