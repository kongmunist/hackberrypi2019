import time
import copy


def init(data):
    data.blocksize = 10
    data.x = 10
    data.y = 10
    data.board = []
    data.gameOn = True

    for i in range(data.x):
        data.board.append([])
        for j in range(data.y):
            data.board[i].append(False)

    data.board[5][5] = True
    data.board[5][4] = True
    data.board[5][3] = True
    data.board[4][5] = True
    data.board[3][4] = True

def main():
    class Struct(object): pass
    data = Struct()
    init(data)


    now = time.time()
    while True:
        if (time.time()-now) > 0.5:
            now = time.time()



            # Updating board simultaneously
            tmpBoard = copy.deepcopy(data.board)

            for i in range(len(data.board)):
                for j in range(len(data.board[0])):
                    neighbors = 0
                    for x1 in [-1, 0, 1]:
                        for y1 in [-1, 0, 1]:
                            try:
                                if data.board[i + x1][j + y1]:
                                    neighbors += 1
                            except:
                                pass

                    if data.board[i][j]:
                        neighbors -= 1  # to account for it counting itself
                        if neighbors < 2:
                            tmpBoard[i][j] = False
                        elif neighbors > 3:
                            tmpBoard[i][j] = False
                        else:
                            tmpBoard[i][j] = True
                    else:
                        if neighbors == 3:
                            tmpBoard[i][j] = True
            data.board = tmpBoard

            # "drawing" it
            print("\n\n\n\n\n")
            for i in range(len(data.board)):
                for j in range(len(data.board[0])):
                    if data.board[i][j]:
                        print(u'\u25A0', end="")
                    else:
                        print(u'\u25A1', end="")
                print("\n")



main()