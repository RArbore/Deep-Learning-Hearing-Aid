import matplotlib.pyplot as plt

path = "trial21/copy_at_7139.txt"

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