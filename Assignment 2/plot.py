import matplotlib.pyplot as plt


def plot_data(data, name, file):
    plt.figure()
    plt.plot(range(len(data)), data[:,0], label="Training Set")
    plt.plot(range(len(data)), data[:,1], label="Validation Set")
    plt.plot(range(len(data)), data[:,2], label="Test Set")
    plt.legend(loc='best')
    plt.title("%s History" %(name))
    plt.xlabel("Epoch")
    plt.ylabel("%s" %(name))
    plt.grid()
    plt.savefig('results/%s' %(file), bbox_inches='tight')
    plt.close()

def visualize_weights(w, file):
    fig, axes = plt.subplots(nrows=2, ncols=5)
    fig.suptitle('Weight Visualization', size=20)
    for i, ax in enumerate(axes.flat):
        heatmap = ax.imshow(w[:,i].reshape((28,28)), cmap = plt.cm.coolwarm)
        ax.set_title(i)
        ax.set_axis_off()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(heatmap, cax=cbar_ax)
    plt.savefig('results/%s' %(file), bbox_inches='tight')
    plt.close()
