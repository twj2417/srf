# import numpy as np
# import scipy
#
# import srfnef as nef
#
#
# # def resize_image(image,unit_size_recon,unit_size_phantom):
# #     return scipy.misc.imresize(image,unit_size_phantom/unit_size_recon)
#
# # def cut_image(image):
# #     _shape = image._shape[0]
# #     return image[ ]
#
# def get_mask(phantom_image: nef.Image, recon_image: nef.Image):
#     unit_size_phantom = phantom_image.size / phantom_image.data.shape
#     unit_size_recon = recon_image.size / recon_image.data.shape
#     resize_image = scipy.misc.imresize(phantom_image, unit_size_recon / unit_size_phantom)
#     mask = np.zeros_like(recon_image)
#
#     return
