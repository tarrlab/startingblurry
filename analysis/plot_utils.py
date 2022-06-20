import matplotlib.pyplot as plt

def set_all_font_sizes(fs):
    
    plt.rc('font', size=fs)          # controls default text sizes
    plt.rc('axes', titlesize=fs)     # fontsize of the axes title
    plt.rc('axes', labelsize=fs)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fs)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fs)    # fontsize of the tick labels
    plt.rc('legend', fontsize=fs)    # legend fontsize
    plt.rc('figure', titlesize=fs)  # fontsize of the figure title
