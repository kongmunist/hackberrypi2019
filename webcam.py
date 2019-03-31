import cv2 as cv
import numpy as np
import torch.nn.functional as F
import torch
import time
from PIL import Image,ImageFilter

class ConvNet2(torch.nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        kern = 4
        self.conv = torch.nn.Conv2d(1,1,kernel_size=kern)
        self.rl = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)

        # Pro config 2
        self.linear1 = torch.nn.Linear(((78-kern+1)//2)**2, 300)
        self.dropout = torch.nn.Dropout(p=0.25)  # try a middle layer of ~30 instead?
        self.linear2 = torch.nn.Linear(300, 50)
        self.linear3 = torch.nn.Linear(50, 5)



    def forward(self,x):
        x = self.conv(torch.tensor([[x]]).to("cpu", dtype=torch.float32))
        x = self.rl(x)
        x = self.pool(x)

        x = x.view(-1,x.numel())

        # config 2
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        kern = 5
        self.conv = torch.nn.Conv2d(1,1,kernel_size=kern)
        self.rl = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)

        # Pro config 2
        self.linear1 = torch.nn.Linear(((78-kern+1)//2)**2, 300)
        self.dropout = torch.nn.Dropout(p=0.25)  # try a middle layer of ~30 instead?
        self.linear2 = torch.nn.Linear(300, 50)
        self.linear3 = torch.nn.Linear(50, 5)



    def forward(self,x):
        x = self.conv(torch.tensor([[x]]).to("cpu", dtype=torch.float32))
        x = self.rl(x)
        x = self.pool(x)

        x = x.view(-1,x.numel())

        # config 2
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

def editPic(img,size=80):
    try:
        img = img.convert("L").resize((size,size)).filter(ImageFilter.FIND_EDGES).crop((1,1,size-1,size-1))
    except:
        pass
    return img




def main():
    cap = cv.VideoCapture(0)


    thirteen1 = ConvNet()
    thirteen1.load_state_dict(torch.load("13.pickle"))
    thirteen1.eval()

    thirteen2 = ConvNet()
    thirteen2.load_state_dict(torch.load("13also"))
    thirteen2.eval()


    thirteen4 = ConvNet2()
    thirteen4.load_state_dict(torch.load("notgreat"))
    thirteen4.eval()


    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        b = Image.fromarray(frame).crop((280,0,1000,720))
        a = b.convert("L").resize((80,80)).filter(ImageFilter.FIND_EDGES).crop((1,1,80-1,80-1))
        a = np.array(a)
        for i in range(len(a)):
            for j in range(len(a[0])):
                if a[i][j] < 60:
                    a[i][j] = 0

        eval1 = thirteen1(a)
        eval2 = thirteen2(a)
        eval4 = thirteen4(a)

        print(eval1.max(1)[1].item()+1,eval2.max(1)[1].item()+1,eval4.max(1)[1].item()+1)
        print()
        # time.sleep(.3)



        b = b.resize((400,400))#.filter(ImageFilter.FIND_EDGES)
        # Display the resulting frame
        cv.imshow('frame',np.array(b)
                  )
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

main()

