# Updated Animation Starter Code

from tkinter import *
import copy

####################################
# customize these functions
####################################

def init(data):
    data.blocksize = 10
    data.x = data.width//data.blocksize
    data.y = data.height//data.blocksize
    data.board = []
    data.gameOn = True


    for i in range(data.x):
        data.board.append([])
        for j in range(data.y):
            data.board[i].append(False)



    # load data.xyz as appropriate


def mousePressed(event, data):
    data.board[event.x//10][event.y//10] = not (data.board[event.x//10][event.y//10])



def keyPressed(event, data):
    # print(event.keysym)
    if event.keysym == "space":
        data.gameOn = not data.gameOn
    # use event.char and event.keysym
    pass

def timerFired(data):
    if data.gameOn:
        tmpBoard = copy.deepcopy(data.board)

        for i in range(len(data.board)):
            for j in range(len(data.board[0])):
                neighbors = 0
                for x1 in [-1,0,1]:
                    for y1 in [-1, 0 ,1]:
                        try:
                            if data.board[i+x1][j+y1]:
                                neighbors += 1
                        except:
                            pass
                # if neighbors != 0:
                #     print(neighbors)

                if data.board[i][j]:
                    neighbors -= 1    # to account for it counting itself
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



    pass

def redrawAll(canvas, data):
    if data.gameOn:
        canvas.create_text(5,5,text="time ON", anchor=NW)
    else:
        canvas.create_text(5,5,text="time OFF", anchor=NW)


    for i in range(len(data.board)):
        for j in range(len(data.board[0])):
            if data.board[i][j]:
                canvas.create_rectangle(i*data.blocksize,j*data.blocksize,(i+1)*data.blocksize,(j+1)*data.blocksize, fill = 'red')


    # draw in canvas


####################################
# use the run function as-is
####################################

def run(width=300, height=300):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 400 # milliseconds
    root = Tk()
    root.resizable(width=False, height=False) # prevents resizing window
    init(data)
    # create the root and the canvas
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

run(400, 200)