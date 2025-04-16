from matplotlib import pyplot as plt
import tensorflow as tf

assert tf.__version__.startswith('2')
import numpy as np
import segmentation_models as sm
from PIL import Image
from os import listdir

sm.set_framework('tf.keras')
sm.framework()

rad = 1225

last_added_thing = None


def mouse_event(event):
    global last_added_thing
    if last_added_thing:
        last_added_thing.remove()
    print('x: {} and y: {}'.format(event.xdata, event.ydata))
    last_added_thing = plt.Circle((event.xdata, event.ydata), rad , alpha=0.1, color='red') #ax.scatter(event.xdata, event.ydata, c='g', marker='o')#plot semitransparent circle of certain radius with center at chosen location
    plt.gca().add_artist(last_added_thing)
    plt.draw()

dataDirectory = r"/Users/anna.lisitsyna/Documents/Corrosion estimation/Pics"
savingDirectory = r"/Users/anna.lisitsyna/Documents/Corrosion estimation/Pics" #r"/Volumes/USB DISK"

filenames_scanner = []
not_to_use = []
directoriesScanner = [r"/20220617/scanner/before/",
                           r"/20220719/scanner/before/",
                           r"/20220823/scanner/before/",
                           r"/20220906/scanner/before/",
                           r"/20220919/scanner/before/",
                           r"/20230210/scanner/before/"
                           ]

for i in range(len(directoriesScanner)):
    filenames_temp = list(sorted(listdir(dataDirectory + directoriesScanner[i])))
    filenames_temp = [filenames_temp[ind_temp] for ind_temp in range(len(filenames_temp)) if
                      dataDirectory + directoriesScanner[i] + filenames_temp[ind_temp] not in not_to_use]
    filenames_temp = [dataDirectory + directoriesScanner[i] + filenames_temp[j] for j in
                   range(len(filenames_temp))]
    print(filenames_temp)
    filenames_scanner.extend(filenames_temp)
#filenames_scanner = ['/Users/anna.lisitsyna/Documents/Corrosion estimation/Pics/20230210/scanner/before/3-6/3-6-1.tiff']
filenames_scanner = [filenames_scanner[i]  for i in range(len(filenames_scanner)) if filenames_scanner[i].split('.')[-1] in ['tif', 'tiff']]
print(filenames_scanner)

for filename_temp in filenames_scanner:
    #print(filename_temp)
    #print(filename_temp.split('.')[-2])
    scanner_temp = Image.open(filename_temp)
    scanner_temp = np.array(scanner_temp)

    fig, ax = plt.subplots()
    plt.title(filename_temp.split('/')[-4]+'/'+filename_temp.split('/')[-3]+'/'+filename_temp.split('/')[-2]+'/'+filename_temp.split('/')[-1])
    cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
    plt.imshow(scanner_temp)
    print(filename_temp)
    plt.close()

plt.show()