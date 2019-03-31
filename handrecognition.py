import torch
import torch.nn.functional as F

from PIL import Image,ExifTags,ImageFilter,ImageOps
import numpy as np
import time


def rotateImg(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif=dict(image._getexif().items())

        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)
        return image
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        return image

def editPic(img,size=80):
    try:
        img = img.convert("L").resize((size,size)).filter(ImageFilter.FIND_EDGES).crop((1,1,size-1,size-1))
    except:
        pass
    return img



def importPics(filename,y,process = False):
    if process:
        pics = []
        finalData = []
        now = time.time()
        numPics = 50

        # Import and edit each image
        for i in range(numPics):
            id = filename + "/IMG_" + str(6636+i) + ".jpg"
            pics.append(Image.open(id))
        pics = [editPic(p) for p in pics]


        # make rotations and np.array them
        for p in pics:
            finalData.append(p)
            # finalData.append(p.rotate(90, expand=True))
            # finalData.append(p.rotate(180, expand=True))
            # finalData.append(p.rotate(270, expand=True))
            # Rotated images for more data
        # finalData = [np.array(f) for f in finalData]

        # print(time.time()-now) #processing takes ~.22 seconds per pic, 50*.22 = 11 secs

        for f in range(len(finalData)):
            finalData[f].save("processedPhots/" + str(f) + ".jpg","JPEG")
    else:
        numPics = 200  #50*4 = 200 total
        pics = []
        num = 6
        count = -1

        # trans = [-20,-10,10,20]
        trans = [-20,20]
        dup = 16  # CHANGE THIS IF YOU EVER MESS WITH THE OPERATIONS BEING DUPLICATED
        for i in range(numPics):
            try:

                pics.append(Image.open("processedPhots/" + str(i) + ".jpg").convert("L")) #.point(lambda p: p > 100 and 255))
                pics[-1] = pics[-1].rotate(270)
                tmp = []
                # pics.append(ImageOps.mirror(pics[-1]))



                # Data Augmentation

                tmp.append(ImageOps.mirror(pics[-1]))
                tmp.append(pics[-1].rotate(10))
                tmp.append(pics[-1].rotate(-10))
                # tmp.append(pics[-1].rotate(-20))
                # tmp.append(pics[-1].rotate(20))
                # tmp.append(tmp[0].rotate(10))
                # tmp.append(tmp[0].rotate(-10))
                # tmp.append(tmp[0].rotate(20))
                # tmp.append(tmp[0].rotate(-20))


                # tmp = []
                for i in trans:
                    for j in trans:
                        tmp.append(pics[-1].transform(pics[-1].size, Image.AFFINE,(1,0,i,0,1,j)))
                        tmp.append(tmp[0].transform(pics[-1].size, Image.AFFINE,(1,0,i,0,1,j)))
                        tmp.append(tmp[1].transform(pics[-1].size, Image.AFFINE,(1,0,i,0,1,j)))
                        # tmp.append(tmp[2].transform(pics[-1].size, Image.AFFINE,(1,0,i,0,1,j)))
                        # tmp.append(tmp[3].transform(pics[-1].size, Image.AFFINE,(1,0,i,0,1,j)))
                        # tmp.append(tmp[4].transform(pics[-1].size, Image.AFFINE,(1,0,i,0,1,j)))

                pics.extend(tmp)

                count = count + 1
                if count % 10 == 0:
                    num = num - 1
                for i in range(dup):
                    y.append(num)

            except:
                pass

        pics = [np.array(p) for p in pics]

        return pics


def rateModel(pics):
    b=[]
    c=[]
    for i in range(len(pics)):
        b.append(curMod(pics[i]))
        c.append(b[i].max(1))

    d = [g[1].item()+1 for g in c]
    tmp = torch.tensor(d) == torch.tensor(y)
    print(tmp.sum(),end="")
    print(tmp.numel())



def testPic(image,show = False):
    curMod.eval()
    if show:
        Image.fromarray(image).show()
    return curMod(image)


def impTestPics(file):
    newIms = []
    for i in range(27):
        img = Image.open("testPhotos/" + str(i) + ".jpg")
        img = img.convert("L")  # .point(lambda p: p > 100 and 255)
        img = np.array(img)
        newIms.append(img)
    return newIms

def testAll(ims):
    curMod.eval()

    b = []
    c = []
    for i in range(len(ims)):
        b.append(curMod(ims[i]))
        c.append(b[i].max(1))

    d = [g[1].item() + 1 for g in c]
    y = [5,5,1,3,3,1,5,3,4,4,2,2,5,4,3,2,1,4,4,2,2,2,2,1,1,5,5]
    tmp = torch.tensor(d) == torch.tensor(y)
    print(tmp.sum(), end="")
    print(tmp.numel())




class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        kern = 5
        self.conv = torch.nn.Conv2d(1,1,kernel_size=kern)
        # self.conv2 = torch.nn.Conv2d(4, 8, kernel_size=kern)
        # self.conv2 = torch.nn.Conv2d(1, 1, kernel_size=kern)
        self.rl = torch.nn.ReLU()
        # self.pool = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)


        # best so far??? IDK they all suck
        # self.linear1 = torch.nn.Linear(1444,200)
        # self.dropout = torch.nn.Dropout(p=0.3) # try a middle layer of ~30 instead?
        # self.linear2 = torch.nn.Linear(200, 5)


        # Pro config 2
        # self.linear1 = torch.nn.Linear(8*((78-kern+1)//2)**2, 100)
        # self.linear1 = torch.nn.Linear(((78-kern+1)//2)**2, 300)
        self.linear1 = torch.nn.Linear(((78-kern+1))**2, 300)
        self.dropout = torch.nn.Dropout(p=0.3)  # try a middle layer of ~30 instead?
        self.linear2 = torch.nn.Linear(300, 50)
        self.linear3 = torch.nn.Linear(50, 5)

        # pro config 3 X bad setup, no improvement
        # self.linear1 = torch.nn.Linear(((78 - kern + 1) // 2) ** 2, 320)
        # self.dropout = torch.nn.Dropout(p=0.25)  # try a middle layer of ~30 instead?
        # self.linear2 = torch.nn.Linear(320, 80)
        # self.linear3 = torch.nn.Linear(80, 20)
        # self.linear4 = torch.nn.Linear(20, 5)


    def forward(self,x):
        x = self.conv(torch.tensor([[x]]).to("cpu", dtype=torch.float32))
        x = F.relu(x)
        # x = self.pool(x)
        x= self.dropout(x)

        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.pool2(x)

        # x = x + torch.tensor(torch.randn(x.size()))
        # adding noise to the input


        # x = x + torch.tensor(torch.randn(x.size()))
        # x = self.dropout(x)

        x = x.view(-1,x.numel())

        # x = self.linear1(x)
        #
        # x = self.linear3(x)

        # config 2
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        #config 3
        # x = F.relu(self.linear1(x))
        # x = self.dropout(x)
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        # x = self.linear4(x)


        return x


y=[]
pics = importPics("Photos",y) # 2448x2448x3 images
testPhotos = impTestPics("testPhotos")

batch = len(y)
inSize = len(pics[0])
outSize = 5
numEpochs = 50

# Constructing the model
curMod = ConvNet()

# Define a loss and optimization function
# loss_fn = torch.nn.MSELoss(reduction='sum')
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(curMod.parameters(), lr=.0001)


# for i in range(len(pics)):
#     p = np.random.rand(len(pics[i]), len(pics[i][0])) * 30 + pics[i]
#     for k in range(len(p)):
#         for j in range(len(p[0])):
#             if p[k][j] < 0:
#                 p[k][j] = 0
#             elif p[k][j] > 255:
#                 p[k][j] = 255
#     pics[i] = p

for i in range(numEpochs):
    curMod.train()
    lossTotal = 0

    for p in range(len(pics)):
        optimizer.zero_grad()

        tmp = pics[p]
        tmp = np.random.rand(len(tmp), len(tmp)) * 40 + tmp
        for k in range(len(tmp)):
            for j in range(len(tmp[0])):
                # print(tmp[k][j])
                if tmp[k][j] < 0:
                    tmp[k][j] = 0
                elif tmp[k][j] > 255:
                    tmp[k][j] = 255
        # print(tmp)



        # outputs = curMod(tmp)
        outputs = curMod(pics[p])
        ans = torch.tensor([[1 if h == y[p] else 0 for h in range(1,6)]]).to("cpu",dtype= torch.long)

        # print(outputs.shape)
        # print(y[p])

        loss = loss_fn(outputs, ans.max(1)[1])

        loss.backward()
        optimizer.step()

        lossTotal += loss.item()
    if (i+1)%1 == 0:
        curMod.eval()
        rateModel(pics)
        testAll(testPhotos)
    print(str(i) + " "  +  str(lossTotal))
    print()




# currentModel(pics[0])