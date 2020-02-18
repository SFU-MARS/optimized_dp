import numpy as np
import matplotlib.pyplot as plt


# Read files
f = open("para_opt_res.txt", "r")


#fig = plt.figure()

for k in [1,2,4,5,10]:
    for j in [1,2,4,5,10]:
        result = []
        for i in [1,2,4,5,10]:
            value = f.readline()
            result.append(float(value))
        plt.bar([1,2,4,5,10], result, align='center', alpha=0.5)
        print(result)
        plt.xlabel('i split ratio')
        plt.ylabel('time')
        plt.ylim(ymax=0.01)
        plt.title('k split ratio = ' + str(k) + ', j split ratio = ' + str(j))
        plt.xticks([1,2,4,5,10])
        file_name = "k" + str(k) + "_j" + str(j) + ".png"
        plt.savefig(file_name)
        plt.clf()
