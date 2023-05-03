import numpy as np

# neg -> pos, pos -> neg
def find_sign_change(grid, V, x):
    indexAtX = grid.get_index(x) # default 2d system
    print("index at x {} ".format(indexAtX))
    valueAtX = V[indexAtX[0],indexAtX[1], :]  # [V[this position at RAS], V[this position at RAS-tstep], ..., V[this position at tau=0]]
    print("valueAtX is {}".format(valueAtX))

    valueIsNeg = (valueAtX<=0).astype(int) # neg = 1, not neg = 0
    print(valueIsNeg)

    checkList = valueIsNeg - np.append(valueIsNeg[1:],valueIsNeg[-1]) # move left 1 position
    # neg -> pos, pos -> neg
    # neg(True) - pos(False) = 1 --> neg to pos
    # pos(False) - neg(True) = -1 --> pos to neg
    return np.where(checkList==1)[0], np.where(checkList==-1)[0]  # return the time point of tau with positive and negative edges