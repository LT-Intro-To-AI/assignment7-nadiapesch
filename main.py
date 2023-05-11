from typing import Tuple
from neural import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def parse_line(line: str) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output (transormfing output as appropriate)

    Args:
        line - one line of the CSV as a string

    Returns:
        tuple of input list and output list
    """
    tokens = line.split("\t")
    # print(tokens)
    out = int(tokens[0])
    # print(out)
    output = [0 if out < 3000000 else 0.5 if out < 5000000 else 1]
    # print(output)

    inpt = [float(x) for x in tokens[1:5]]
    # print(inpt)
    return (inpt, output)


def normalize(data: List[Tuple[List[float], List[float]]]):
    """Makes the data range for each input feature from 0 to 1

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)
    """
    leasts = len(data[0][0]) * [100.0]
    mosts = len(data[0][0]) * [0.0]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            if data[i][0][j] < leasts[j]:
                leasts[j] = data[i][0][j]
            if data[i][0][j] > mosts[j]:
                mosts[j] = data[i][0][j]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            data[i][0][j] = (data[i][0][j] - leasts[j]) / \
                (mosts[j] - leasts[j])
    return data


with open("data.txt", "r") as f:
    f.readline()
    training_data = [parse_line(line)
                     for line in f.readlines() if len(line) > 4]

# for line in training_data:
#     print(line)

td = normalize(training_data)


train_data, test_data = train_test_split(td, test_size=.15)
print(len(train_data))
print(len(test_data))
# for line in td:
#     print(line)

nn = NeuralNet(4, 6, 1)

nn.train(train_data, learning_rate=.85)
result = nn.test_with_expected(test_data)
x, y = [], []
print(x, y)
for i in result:
    print(f"desired: {i[1]}, actual: {i[2]}")
    x += (i[1])
    y += (i[2])
print(x, y)

# print(result)

# make data:
# np.random.seed(3)
# x = 0.5 + np.arange(10)
# y = np.random.uniform(2, 7, len(x))


# # plot
# fig, ax = plt.subplots()

# ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

# ax.set(xlim=(0, 8), xticks=np.arange(0, 10),
#        ylim=(0, 8), yticks=np.arange(0, 10))

# plt.show()


# plt.style.use('_mpl-gallery-nogrid')


# # make data
# x = [1,2]
# colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(x)))

# # plot
# fig, ax = plt.subplots()
# ax.pie(x, colors=colors, radius=3, center=(4, 4),
#        wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)

# ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#        ylim=(0, 8), yticks=np.arange(1, 8))

# plt.show()


plt.style.use('_mpl-gallery')

# make the data
# np.random.seed(3)
# x = 4 + np.random.normal(0, 2, 24)
# y = 4 + np.random.normal(0, 2, len(x))
# size and color:
sizes = np.random.uniform(15, 80, len(x))
colors = np.random.uniform(15, 80, len(x))

# plot
fig, ax = plt.subplots()

ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=1)

ax.set(xlim=(0, 1), xticks=np.arange(0, 1),
       ylim=(0, 1), yticks=np.arange(0, 1))

plt.show()
