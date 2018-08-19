def showims(img_array_list, label_list=None):
    from matplotlib import pyplot as plt
    fig = plt.figure()
    for i, img in enumerate(img_array_list):
        a = fig.add_subplot(1, len(img_array_list), i+1)
        imgplot = plt.imshow(img)
        if label_list is not None:
            a.set_title(label_list[i])
    plt.show()

def saveims(img_array_list, label_list=None, savepath='myim.png'):
    from matplotlib import pyplot as plt
    fig = plt.figure()
    for i, img in enumerate(img_array_list):
        a = fig.add_subplot(1, len(img_array_list), i+1)
        imgplot = plt.imshow(img)
        if label_list is not None:
            a.set_title(label_list[i])
    plt.savefig(savepath)