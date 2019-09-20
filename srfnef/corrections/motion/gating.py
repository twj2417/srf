import numpy as np
from srfnef import nef_class
from srfnef.data import Listmode,Lors
import srfnef as nef


@nef_class
class Gating:
    listmode:Listmode

    def __call__(self):
        codz,bins = self.get_raw_codz_and_bins()
        codz = codz - np.min(codz)
        amplitudes = find_amplitudes(codz)
        listmode_parts = []
        for i in range(8):
            index_in_bins = np.where((codz>=amplitudes[i])&(codz<amplitudes[i+1]))[0]
            if index_in_bins.size>0:
                value = bins[index_in_bins[0]].data
                lors_data = bins[index_in_bins[0]].lors.data
                for k in range(1,index_in_bins.size):
                    value = np.vstack((value,bins[index_in_bins[k]].data))
                    lors_data = np.vstack((lors_data,bins[index_in_bins[k]].lors.data))
            listmode_parts.append(nef.Listmode(value,nef.Lors(lors_data)))
        return listmode_parts

    
    def get_raw_codz_and_bins(self):
        time = self.listmode.data[:,1]
        value = self.listmode.data[:,0]
        centerz = get_centerz(self.listmode.lors.data)
        num_interval = int(np.max(time)/0.1)+1
        codz = np.zeros((num_interval+1,))
        bins = []
        for i in range(num_interval+1):
            index = np.where((time>=i*0.1)&(time<(i+1)*0.1))[0]
            codz[i] = np.sum(centerz[index[0]:index[-1]+1]*value[index[0]:index[-1]+1])/index.size
            bins.append(nef.Listmode(self.listmode.data[index,0].reshape(-1,1),nef.Lors(self.listmode.lors.data[index,:])))
        return codz,bins


def get_centerz(listmode_data):
    fstz = listmode_data[:,2]
    sndz = listmode_data[:,5]
    centerz = (fstz+sndz)/2
    return centerz


def find_amplitudes(codz):
    amplitudes = np.zeros((9,))
    step = np.max(codz)/100
    num_ = np.zeros((100,))
    for i in range(100):
        num_[i] = np.where(codz<=i*step)[0].size/codz.size
    for i in range(1,8):
        amplitudes[i] = np.max(np.where(num_<i/8)[0])*step
    amplitudes[8] = np.max(codz)+0.1
    return amplitudes




