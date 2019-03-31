import time
from PIL import Image,ImageFilter


def editPic(img,size=80):
    try:
        img = img.convert("L").resize((size,size)).filter(ImageFilter.FIND_EDGES).crop((1,1,size-1,size-1))
    except:
        pass
    return img

def importPics(filename,process = False):
    if process:
        pics = []
        finalData = []
        now = time.time()
        numPics = 10

        # Import and edit each image
        for i in range(numPics):
            id = filename + "/IMG_" + str(6719+i) + ".jpg"
            pics.append(Image.open(id))
        pics = [editPic(p) for p in pics]


        # make rotations and np.array them
        for p in pics:
            finalData.append(p)
            # Rotated images can make more data

        # print(time.time()-now) #processing takes ~.22 seconds per pic, 50*.22 = 11 secs

        for f in range(len(finalData)):
            finalData[f].save("testPhotos/" + str(17+f) + ".jpg","JPEG")


importPics("test",True)

