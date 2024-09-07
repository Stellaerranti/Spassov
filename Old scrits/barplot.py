import matplotlib.pyplot as plt
import numpy as np

def plot_barplot(z,M):
    logic = []
    for i in range(1,M.shape[0]):
      if M[i] <= 0 and M[i-1] >  0 :
        logic.append(z[i])
      if M[i] > 0 and M[i-1] <= 0:
        logic.append(z[i])
    if(len(logic) % 2 !=0):
      logic.append(z[M.shape[0]-1])
    logic = np.asarray(logic)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5), gridspec_kw={'width_ratios': [3, 1]})
    ax1.plot(M,z)
    ax1.set_ylim(80,70)
    ax1.set_xlim(-1.2,1.2)
    ax1.axvline(0,c='r',linestyle = 'dashed')
    count = 0
    ax2.fill_between(z,0,logic[0], facecolor='grey',hatch = '/')
    while(count < logic.shape[0]-1):
       ax2.fill_between(z,logic[count],logic[count+1], facecolor='black')
       count = count+2
    ax2.set_ylim(70,80)
    ax2.set_xlim(0,1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.show()
    

depth, polarity = np.loadtxt('Kuldara_polarity for Dima.txt', unpack = True)

plot_barplot(depth, polarity)