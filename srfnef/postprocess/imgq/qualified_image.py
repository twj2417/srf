# # encoding: utf-8
# '''
# @author: Minghao Guo
# @contact: mh.guo0111@gmail.com
# @software: nef
# @file: qualified_image.py
# @date: 4/14/2019
# @desc:
# '''
# import srfnef as nef
# from srfnef import nef_class, float_or_none
# from .index_correction import *
# from .index_recon import *
#
#
# @nef_class
# class QualitifiedImage:
#     image: nef.Image
#     mask: nef.Image
#     hot_cold_ratio: float_or_none
#     num_bg: float_or_none
#     _acc: float_or_none
#     _sd: float_or_none
#     _bv: float_or_none
#     _snr2: float_or_none
#     _noise1: float_or_none
#     _noise2: float_or_none
#     _contrast_hot: float_or_none
#     _contrast_cold: float_or_none
#     _crc1: float_or_none
#     _crc2: float_or_none
#     _snr1: float_or_none
#     _nsd: float_or_none
#     _ai: float_or_none
#     _fwhm_x: float_or_none
#     _fwtm_x: float_or_none
#     _fwhm_y: float_or_none
#     _fwtm_y: float_or_none
#     _fwhm_z: float_or_none
#     _fwtm_z: float_or_none
#     _sd_rc: float_or_none
#     _pos_dev: float_or_none
#     _neg_dev: float_or_none
#
#     def __attrs_post_init__(self):
#         if self.image.data is not None and self.mask.data is not None and self.hot_cold_ratio is not None:
#             object.__setattr__(self, '_ctrst_hot',
#                                contrast_hot(self.mask.data, self.image.data, self.hot_cold_ratio))
#             object.__setattr__(self, '_crc2',
#                                se(self.mask.data, self.image.data, self.hot_cold_ratio))
#
#         if self.image.data is not None and self.mask.data is not None:
#             object.__setattr__(self, '_acc', accuracy(self.mask.data, self.image.data))
#             object.__setattr__(self, '_sd', sd(self.mask.data, self.image.data))
#             object.__setattr__(self, '_bv', bv(self.mask.data, self.image.data))
#             object.__setattr__(self, '_snr1', snr1(self.mask.data, self.image.data))
#             object.__setattr__(self, '_snr2', snr2(self.mask.data, self.image.data))
#             object.__setattr__(self, '_noise1', noise1(self.mask.data, self.image.data))
#             object.__setattr__(self, '_noise2', noise2(self.mask.data, self.image.data))
#             object.__setattr__(self, '_ctrst_cold', contrast_cold(self.mask.data, self.image.data))
#             object.__setattr__(self, '_nsd', nsd(self.mask.data, self.image.data))
#             object.__setattr__(self, '_ai', ai(self.mask.data, self.image.data))
#             # object.__setattr__(self, '_fwhm_x', fwhm_x(self.image))
#             # object.__setattr__(self, '_fwtm_x', fwtm_x(self.image))
#             # object.__setattr__(self, '_fwhm_y', fwhm_y(self.image))
#             # object.__setattr__(self, '_fwtm_y', fwtm_y(self.image))
#             # object.__setattr__(self, '_fwhm_z', fwhm_z(self.image))
#             # object.__setattr__(self, '_fwtm_z', fwtm_z(self.image))
#             # object.__setattr__(self, '_sd_rc', sd_rc(self.mask.data, self.image))
#             object.__setattr__(self, '_pos_dev',
#                                positive_deviation(self.mask.data, self.image.data))
#             object.__setattr__(self, '_neg_dev',
#                                negative_deviation(self.mask.data, self.image.data))
#
#         if self.image.data is not None and self.mask.data is not None and self.num_bg is not None:
#             object.__setattr__(self, '_se', se(self.mask.data, self.image.data, self.num_bg))
#
#     def update_metrics(self):
#         self.__attrs_post_init__()
