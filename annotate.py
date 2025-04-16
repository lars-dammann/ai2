from matplotlib import pyplot as plt
#import tensorflow as tf

#assert tf.__version__.startswith('2')
import numpy as np
#import segmentation_models as sm
from PIL import Image
from os import listdir
import pandas as pd
import json

#sm.set_framework('tf.keras')
#sm.framework()

rad = 670

last_added_thing = None


def mouse_event(event):
    global last_added_thing
    if last_added_thing:
        last_added_thing.remove()
    print('x: {} and y: {}'.format(event.xdata, event.ydata))
    last_added_thing = plt.Circle((event.xdata, event.ydata), rad , alpha=0.1, color='red') #ax.scatter(event.xdata, event.ydata, c='g', marker='o')#plot semitransparent circle of certain radius with center at chosen location
    plt.gca().add_artist(last_added_thing)
    plt.draw()

filenames_to_annotate = ['/Volumes/Expansion/Corrosion estimation/Data/New test data/#3-3/Before/#3-3-1/Profilometer picture/1.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#3-3/Before/#3-3-1/Profilometer picture/2.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#3-3/Before/#3-3-1/Profilometer picture/3.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#3-3/Before/#3-3-1/Profilometer picture/4.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#3-3/Before/#3-3-2/Profilometer picture/1.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#3-3/Before/#3-3-2/Profilometer picture/2.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#3-3/Before/#3-3-2/Profilometer picture/3.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#3-3/Before/#3-3-2/Profilometer picture/4.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#3-3/Before/#3-3-3/Profilometer picture/1.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#3-3/Before/#3-3-3/Profilometer picture/2.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#3-3/Before/#3-3-3/Profilometer picture/3.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#3-3/Before/#3-3-3/Profilometer picture/4.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#3-3/Before/#3-3-4 NaCl/Profilometer picture/1.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#3-3/Before/#3-3-4 NaCl/Profilometer picture/2.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#3-3/Before/#3-3-4 NaCl/Profilometer picture/3.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#3-3/Before/#3-3-4 NaCl/Profilometer picture/4.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-11/Before/#4-11-7/Profilometer picture/1.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-11/Before/#4-11-7/Profilometer picture/2.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-11/Before/#4-11-7/Profilometer picture/3.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-11/Before/#4-11-7/Profilometer picture/4.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-11/Before/#4-11-11/Profilometer picture/1.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-11/Before/#4-11-11/Profilometer picture/2.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-11/Before/#4-11-11/Profilometer picture/3.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-11/Before/#4-11-11/Profilometer picture/4.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-11/Before/#4-11-12/Profilometer picture/1.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-11/Before/#4-11-12/Profilometer picture/2.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-11/Before/#4-11-12/Profilometer picture/3.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-11/Before/#4-11-12/Profilometer picture/4.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-7/Profilometer picture/1.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-7/Profilometer picture/2.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-7/Profilometer picture/3.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-7/Profilometer picture/4.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-8/Profilometer picture/1.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-8/Profilometer picture/2.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-8/Profilometer picture/3.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-8/Profilometer picture/4.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-9/Profilometer picture/1.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-9/Profilometer picture/2.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-9/Profilometer picture/3.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-9/Profilometer picture/4.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-10/Profilometer picture/1.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-10/Profilometer picture/2.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-10/Profilometer picture/3.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-10/Profilometer picture/4.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-11/Profilometer picture/1.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-11/Profilometer picture/2.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-11/Profilometer picture/3.png', '/Volumes/Expansion/Corrosion estimation/Data/New test data/#4-4/Before/#4-4-11/Profilometer picture/4.png']
#['/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-7/1.png',
 #'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-7/2.png',
 #'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-7/3.png',
 #'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-7/4.png',
#'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-8/1.png',
 #'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-8/2.png',
 #'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-8/3.png',
 #'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-8/4.png',
#'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-9/1.png',
 #'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-9/2.png',
 #'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-9/3.png',
 #'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-9/4.png',
#'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-10/1.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-10/2.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-10/3.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-10/4.png',
#'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-11/1.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-11/2.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-11/3.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-11/4.png',
#'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/1-6-10/1.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/1-6-10/2.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/1-6-10/3.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/1-6-10/4.png'

# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-7/1.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-7/2.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-7/3.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-7/4.png',
#'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-8/1.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-8/2.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-8/3.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-8/4.png',
#'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-9/1.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-9/2.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-9/3.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-9/4.png',
#'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-10/1.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-10/2.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-10/3.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-10/4.png',
#'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-11/1.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-11/2.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-11/3.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-11/4.png',
#'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-12/1.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-12/2.png',
# '/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-12/3.png',
 #'/Volumes/Expansion/AI2 results 20240119/3-1 with results/before/profilometer/3-1-12/4.png',
    
#'/Volumes/Expansion/AI2 results 20240119/1-6/before/profilometer/1-6-10/1.png',
# '/Volumes/Expansion/AI2 results 20240119/1-6/before/profilometer/1-6-10/2.png',
# '/Volumes/Expansion/AI2 results 20240119/1-6/before/profilometer/1-6-10/3.png',
# '/Volumes/Expansion/AI2 results 20240119/1-6/before/profilometer/1-6-10/4.png'
#]

c1_add = {}

for filename_temp in filenames_to_annotate:
    print("HI!!!! {}".format(filename_temp[:-4]))
    try:
        if filename_temp[-3:]=='png':
            print("I am in annotating png")
            img_temp = Image.open(filename_temp)
            img_temp = np.array(img_temp)
        else:
            img_temp = pd.read_csv(filepath_or_buffer=filename_temp, skiprows=22, engine='python').to_numpy()
        fig, ax = plt.subplots()
        plt.title(filename_temp.split('/')[-4]+'/'+filename_temp.split('/')[-3]+'/'+filename_temp.split('/')[-2]+'/'+filename_temp.split('/')[-1])
        cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
        plt.imshow(img_temp)
        plt.show(block=True)

        c1_add[filename_temp] =  (int(np.round(last_added_thing.center[0])), int(np.round(last_added_thing.center[1])))
        #plt.close()
    except Exception as e:
        print(e)
        print("In file {}".format(filename_temp))

print(json.dumps(c1_add))
print("I am here")

#plt.show()



