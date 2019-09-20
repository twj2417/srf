import sys

import numpy as np

from srfnef import nef_class
from srfnef.data import Image, Umap, Listmode, Lors
from srfnef.geometry import PetEcatScanner
import srfnef as nef

from .listmode2sinogram import lm2sino, sino2lm, get_all_lors_id
from .scatter_fraction import get_scatter_fraction
from .preprocess import pre_all_scatter_position, pre_atten, pre_lors_atten, pre_sumup_of_emission, \
    get_all_crystal_position

np.seterr(divide = 'ignore', invalid = 'ignore')


@nef_class
class ScatterCorrect:
    low_energy: float
    high_energy: float
    resolution: float
    scatter_fraction: float
     
    def __call__(self, emission_image: Image, u_map: Umap, scanner: PetEcatScanner,outdir,out_append=None) -> Listmode:
        if out_append==None:
            out_append = ''
        lors = np.load(outdir+f'/lors{out_append}.npy')
        sinogram = np.load(outdir+f'/sinogram{out_append}.npy')
        index = np.load(outdir+f'/index{out_append}.npy')
        scatter_position = np.load(outdir+f'/scatter_position{out_append}.npy')
        crystal_position = np.load(outdir+f'/crystal_position{out_append}.npy')
        sumup_emission = np.load(outdir+f'/sumup_emission{out_append}.npy')
        atten = np.load(outdir+f'/atten{out_append}.npy')
        atten_lors = np.load(outdir+f'/atten_lors{out_append}.npy')  
        if isinstance(scanner, nef.PetEcatScanner):
            nb_blocks_per_ring = scanner.nb_blocks_per_ring
            nb_crystals_per_ring = scanner.nb_blocks_per_ring * scanner.blocks.shape[1]
            grid_block = np.array(scanner.blocks.shape, dtype = np.int32)
            size_block = np.array(scanner.blocks.size,dtype = np.float32)
        elif isinstance(scanner, nef.PetCylindricalScanner):
            nb_blocks_per_ring = scanner.nb_rsector
            nb_crystals_per_ring = scanner.nb_rsector * scanner.nb_module[0] *scanner.nb_submodule[0] *scanner.nb_crystal[0]
            grid_block = np.array([1,scanner.nb_module[0]*scanner.nb_submodule[0]*scanner.nb_crystal[0],
                                scanner.nb_module[1]*scanner.nb_submodule[1]*scanner.nb_crystal[1]], dtype = np.int32)
            size_block = np.array([20,scanner.mv_module[0]*scanner.nb_module[0],scanner.mv_module[1]*scanner.nb_module[1]],dtype = np.float32)   
        fraction, scale = get_scatter_fraction(emission_image, u_map, index, lors, nb_blocks_per_ring,nb_crystals_per_ring,
                                                      grid_block,size_block,self.low_energy, self.high_energy,
                                                      self.resolution,scatter_position,crystal_position,
                                                      sumup_emission,atten)
        corrected_sinogram = np.zeros_like(sinogram)
        corrected_sinogram[index] = (sinogram[index] - fraction / np.sum(fraction) * np.sum(
            sinogram[index]) * self.scatter_fraction) * scale / (atten_lors + sys.float_info.min)
        corrected_data = sino2lm(scanner, corrected_sinogram, lors)
        return Listmode(corrected_data[:, 6].astype(np.float32), Lors(corrected_data.astype(np.float32)))

def scatter_preprocess(scanner,listmode,emission_image,umap,outdir,out_append=None):
    lors = get_all_lors_id(scanner.nb_crystals)
    sinogram = lm2sino(listmode, scanner)
    index = np.where(sinogram > 0)[0].astype(np.int32)
    scatter_position = pre_all_scatter_position(emission_image)
    crystal_position = get_all_crystal_position(scanner)
    sumup_emission = pre_sumup_of_emission(emission_image, crystal_position, scatter_position)
    atten = pre_atten(umap, crystal_position, scatter_position)
    atten_lors = pre_lors_atten(umap, crystal_position[lors[index, 0]],crystal_position[lors[index, 1]]).reshape(-1, 1)
    np.save(outdir+f'/lors{out_append}.npy',lors)
    np.save(outdir+f'/sinogram{out_append}.npy',sinogram)
    np.save(outdir+f'/index{out_append}.npy',index)
    np.save(outdir+f'/scatter_position{out_append}.npy',scatter_position)
    np.save(outdir+f'/crystal_position{out_append}.npy',crystal_position)
    np.save(outdir+f'/sumup_emission{out_append}.npy',sumup_emission)
    np.save(outdir+f'/atten{out_append}.npy',atten)
    np.save(outdir+f'/atten_lors{out_append}.npy',atten_lors)