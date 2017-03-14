import os
import numpy as np
import matplotlib.pyplot as plt
import pdb

# reading all the file

# for openign error and acc
def open_file(filepath):
    f=open(filepath,'r')
    epochs = list()
    losses = list()
    for line in f:
        _epoch,_loss = line.split(",")
        epochs.append(float(_epoch))
        losses.append(float(_loss))
    return losses


def multi_plot_result(filelist,valid=False,_title=None):
    result_dict = {}
    fig_1 = plt.figure(figsize=(8, 4))
    scale = 5
    if _title:
        print(_title)
        plt.title(_title)
    for filetup in filelist:
        filename,label=filetup
        result_dict[filename]=open_file(filename)
        if valid:
            plt.plot(np.arange(4,121,scale),result_dict[filename],label=label)
        else:
            plt.plot(np.arange(100,140001,100),result_dict[filename],label=label)
        lgd=plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.show()
    plt.savefig(_title+".png",bbox_extra_artists=(lgd,), bbox_inches='tight')



filelist = [("train_10000sen_1-1layers_100units_question2_layer_1_1_NO_ATTN.log","1x1"),
            ("train_10000sen_2-2layers_100units_question2_NO_ATTN.log","2x2"),
            ("train_10000sen_1-5layers_100units_question2_NO_ATTN.log","1x5"),
            ]
        
                
multi_plot_result(filelist,False,"question2")