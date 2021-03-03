import matplotlib.pyplot as plt

def images1(image_label_ds,figuresize=(8,8)):
    plt.figure(figsize=figuresize)
    for n,image_label in enumerate(image_label_ds.take(4)):
        plt.subplot(2,2,n+1)
        plt.imshow(image_label[0])
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('image'+str(n))
    plt.show()
