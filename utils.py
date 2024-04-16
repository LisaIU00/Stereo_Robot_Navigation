from matplotlib import pyplot as plt
import argparse

plt.rcParams["figure.figsize"] = (15,6)

# display image
def imshow(wname,title, img):
    plt.figure(wname); 
    plt.clf()
    plt.imshow(img)
    plt.title(title)
    plt.pause(0.0001)

def plotgraph(title, x, y, xlabel, ylabel, nomeimg,color='m'):
    plt.figure(title)
    plt.plot(x, y,color, linewidth=3)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(nomeimg)
    plt.show(block=False)


def getParams():
    parser = argparse.ArgumentParser(prog='CVproject',description='Stereo Robot Navigation')
    parser.add_argument('-d','--numDisparities',default='128',help='numDisparities parameter for disparity map algorithm',type=int)
    parser.add_argument('-b','--blockSize',default='33',help='blocksize parameter for disparity map algorithm', type=int)
    return parser.parse_args()
