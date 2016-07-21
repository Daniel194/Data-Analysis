from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')


def create_database(hm, variance, step=2, correlation=False, ):
    val = 1
    ys = []

    for i in range(hm):
        y = val + random.randrange(-variance, + variance)
        ys.append(y)

        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= val

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercepte(xs, ys):
    m = (mean(xs) * mean(ys) - mean(xs * ys)) / (mean(xs) ** 2 - mean(xs ** 2))
    b = mean(ys) - m * mean(xs)

    return m, b


def square_error(ys_origin, ys_line):
    return sum((ys_line - ys_origin) ** 2)


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    square_error_regr = square_error(ys_orig, ys_line)
    square_error_y_mean = square_error(ys_orig, y_mean_line)

    return 1 - (square_error_regr / square_error_y_mean)


xs, ys = create_database(40, 40, 2, correlation='pos')

m, b = best_fit_slope_and_intercepte(xs, ys)

# y = m * x + b
regression_line = [m * x + b for x in xs]

r_square = coefficient_of_determination(ys, regression_line)

print(r_square)

predict_x = 8
predict_y = m * predict_x + b

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g', s=100)
plt.plot(xs, regression_line)
plt.show()
