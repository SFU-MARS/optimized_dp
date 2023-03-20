import numpy as np

# neg -> pos, pos -> neg
def find_sign_change(grid, V, x):
    indexAtX = grid.get_index(x)

    valueAtX = V[indexAtX[0],indexAtX[1],indexAtX[2],:]

    valueIsNeg = (valueAtX<=0).astype(int)

    checkList = valueIsNeg - np.append(valueIsNeg[1:],valueIsNeg[-1])
    # neg -> pos, pos -> neg
    # neg(True) - pos(False) = 1 --> neg to pos
    # pos(False) - neg(True) = -1 --> pos to neg
    return np.where(checkList==1)[0], np.where(checkList==-1)[0]