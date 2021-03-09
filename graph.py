import matplotlib.pyplot as plt
import sys

path = "trial"+sys.argv[1]+"/during_training_performance.txt"

f = open(path)
read = f.read()
splits = read.split("\n")
splits.remove("")

index = []
value = []
for s in splits:
    split = s.split(" ")
    index.append(int(split[0]))
    value.append(float(split[1]))

plt.plot(index, value)
plt.show()