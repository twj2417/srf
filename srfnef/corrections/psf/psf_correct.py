# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: psf_correct.py
@date: 5/10/2019
@desc:
'''
from srfnef import nef_class, List
from srfnef.data import Image
from .psf_fit import PsfFit
from .deconvolute import Deconvolute


@nef_class
class PsfCorrect:
    pnt_img_path: str
    deconv: Deconvolute

    def __call__(self, image: Image, n_iter: int = 10):
        fitter = PsfFit(5, 1000)
        fitx, fity, fitz = fitter(self.pnt_img_path)
        if self.deconv is None:
            object.__setattr__(self, 'deconv', Deconvolute(n_iter, fitx, fity, fitz))
            self.deconv.make_kernel_xy(image)
            self.deconv.make_kernel_z(image)
        else:
            object.__setattr__(self, 'deconv', self.deconv.update(n_iter = n_iter))
        return self.deconv(image)
