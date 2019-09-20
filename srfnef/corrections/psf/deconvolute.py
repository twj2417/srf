# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: srf_ct
@file: deconvolute.py
@date: 3/31/2019
@desc:
'''

import numpy as np
from scipy import sparse
from scipy import interpolate

import srfnef as nef
from srfnef import nef_class
from .fitted_x import FittedX
from .fitted_y import FittedY
from .fitted_z import FittedZ
from srfnef.data import Image
from srfnef.utils import tqdm
from scipy.sparse import coo_matrix
import attr
import tensorflow as tf


@nef_class
class Deconvolute:
    n_iter: int
    fitted_x: FittedX
    fitted_y: FittedY
    fitted_z: FittedZ
    kernel_xy: coo_matrix = attr.ib(default = None)
    kernel_z: coo_matrix = attr.ib(default = None)

    def __call__(self, image: Image):
        if self.kernel_xy is None:
            raise ValueError('Please do make kernel first')
        from srfnef.utils import declare_eager_execution
        declare_eager_execution()

        x = np.ones((image.shape[0] * image.shape[1], image.shape[2]), dtype = np.float32)
        x_tf = tf.Variable(x)

        d = image.data.reshape((-1, image.shape[2]))
        d_tf = tf.constant(d)
        kernel_xy_tf = tf.sparse.SparseTensor(indices = list(zip(self.kernel_xy.row,
                                                                 self.kernel_xy.col)),
                                              values = self.kernel_xy.data,
                                              dense_shape = self.kernel_xy.shape)
        kernel_z_tf = tf.sparse.SparseTensor(indices = list(zip(self.kernel_z.row,
                                                                self.kernel_z.col)),
                                             values = self.kernel_z.data,
                                             dense_shape = self.kernel_z.shape)
        if self.kernel_xy.nnz * image.shape[2] > 2 ** 31:
            raise ValueError(
                'Cannot use GPU when output.shape[1] * nnz(a) > 2^31 [Op:SparseTensorDenseMatMul]')
        for _ in tqdm(range(self.n_iter)):
            c_tf = tf.transpose(tf.sparse.sparse_dense_matmul(kernel_z_tf, x_tf, adjoint_b = True))
            c_tf = tf.sparse.sparse_dense_matmul(kernel_xy_tf, c_tf) + 1e-16
            c_tf = d_tf / c_tf
            c_tf = tf.sparse.sparse_dense_matmul(kernel_xy_tf, c_tf, adjoint_a = True)
            c_tf = tf.transpose(tf.sparse.sparse_dense_matmul(kernel_z_tf, c_tf,
                                                              adjoint_a = True,
                                                              adjoint_b = True))
            x_tf = x_tf * c_tf
        image = image.update(data = x_tf.numpy().reshape(image.shape))
        return image

    def make_kernel_xy(self, image: Image):
        if self.kernel_xy is not None:
            return self.kernel_xy
        kernel_xy = self.xy_main(image, half_patch_range = 5)
        object.__setattr__(self, 'kernel_xy', kernel_xy)

    def make_kernel_z(self, image: Image):
        if self.kernel_z is not None:
            return self.kernel_z
        kernel_z = self.z_main(image)
        object.__setattr__(self, 'kernel_z', kernel_z)

    def make_polargrid(self, xmesh, ymesh):
        rmesh = np.sqrt(xmesh ** 2 + ymesh ** 2)
        pmesh = np.arctan2(ymesh, xmesh)
        return rmesh, pmesh

    def locate_kernel_xy(self, x, y, sigx0, sigx1, sigy):
        ans0 = 1.0 / np.sqrt(2 * np.pi * sigx0 ** 2) \
               / np.sqrt(2 * np.pi * sigy ** 2) \
               * np.exp(-x ** 2 / 2 / sigx0 ** 2) \
               * np.exp(-y ** 2 / 2 / sigy ** 2)
        ans0[x > 0] = 0.0
        ans1 = 1.0 / np.sqrt(2 * np.pi * sigx1 ** 2) \
               / np.sqrt(2 * np.pi * sigy ** 2) \
               * np.exp(-x ** 2 / 2 / sigx1 ** 2) \
               * np.exp(-y ** 2 / 2 / sigy ** 2)
        ans1[x <= 0] = 0.0
        ans = ans0 + ans1
        return ans / np.sum(ans)

    def locate_kernel_xy_spot(self, x, y, sigx0, sigx1, sigy):
        if x < 0:
            return 1.0 / np.sqrt(2 * np.pi * sigx0 ** 2) \
                   / np.sqrt(2 * np.pi * sigy ** 2) \
                   * np.exp(-x ** 2 / 2 / sigx0 ** 2) \
                   * np.exp(-y ** 2 / 2 / sigy ** 2)
        else:
            return 1.0 / np.sqrt(2 * np.pi * sigx1 ** 2) \
                   / np.sqrt(2 * np.pi * sigy ** 2) \
                   * np.exp(-x ** 2 / 2 / sigx1 ** 2) \
                   * np.exp(-y ** 2 / 2 / sigy ** 2)

    def rotate_kernel(self, img, angle):
        '''
        rotate an 2D kernel in X-Y plane.
        Args:
            img: input kernel to be rotated
            angle: the rotation angle
        Returns:
            rotated kernel
        '''
        from scipy import ndimage
        img_r = ndimage.interpolation.rotate(img, np.rad2deg(angle), reshape = False)
        return img_r

    def compensate_kernel(self, kernel_array, factor, xrange):
        '''
        The experimental xy_kernel parameters were corase along the x axis.
        Interpolate the kernel parameters in the whole range.

        Args:
            kernel_array: kerenl parameters to be interpolated.
            factor: scale ratio to be refined.
            xrange: the x axis range of kernel.
        Returns:
            An interpolated kernel parameter array.
        '''
        from scipy import interpolate
        nb_samples = len(kernel_array)
        nb_new_samples = int(nb_samples * factor)
        x = np.linspace(0, xrange, nb_samples)
        nb_columns = kernel_array.shape[1]

        kernel_new = np.zeros([nb_new_samples, nb_columns])
        for i_column in range(nb_columns):
            y = kernel_array[:, i_column]
            f = interpolate.interp1d(x, y)
            xnew = np.linspace(0, xrange, nb_new_samples)
            kernel_new[:, i_column] = f(xnew)
        return kernel_new

    def xy_main(self, image: Image, *, half_patch_range = 0):
        x = (np.arange(image.shape[0]) + 0.5) * image.unit_size[0] - image.size[0] / 2 + \
            image.center[0]
        y = (np.arange(image.shape[1]) + 0.5) * image.unit_size[1] - image.size[1] / 2 + \
            image.center[1]
        xmesh, ymesh = np.meshgrid(x, y, indexing = 'ij')
        rmesh, pmesh = self.make_polargrid(xmesh, ymesh)

        fsigx0 = interpolate.interp1d(self.fitted_x.ux, self.fitted_x.sigx0,
                                      fill_value = 'extrapolate')
        fsigx1 = interpolate.interp1d(self.fitted_x.ux, self.fitted_x.sigx1,
                                      fill_value = 'extrapolate')
        fsigy = interpolate.interp1d(self.fitted_x.ux, self.fitted_y.sigy,
                                     fill_value = 'extrapolate')
        out = sparse.lil_matrix((image.shape[0] * image.shape[1],
                                 image.shape[0] * image.shape[1]),
                                dtype = np.float32)
        for i in nef.utils.tqdm(range(image.shape[0])):
            for j in range(image.shape[1]):
                irow = i * image.shape[1] + j
                r, phi = rmesh[i, j], pmesh[i, j]
                if r > np.max(self.fitted_x.ux):
                    continue

                px = np.arange(max(i - half_patch_range, 0),
                               min(i + half_patch_range + 1, image.shape[0]))
                py = np.arange(max(j - half_patch_range, 0),
                               min(j + half_patch_range + 1, image.shape[1]))
                xs0 = (px + 0.5) * image.unit_size[0] - image.size[0] / 2 + image.center[0]
                ys0 = (py + 0.5) * image.unit_size[1] - image.size[1] / 2 + image.center[1]
                xs0, ys0 = np.meshgrid(xs0, ys0, indexing = 'ij')
                xs = xs0 * np.cos(phi) + ys0 * np.sin(phi)
                ys = -xs0 * np.sin(phi) + ys0 * np.cos(phi)
                wei = self.locate_kernel_xy(xs - r, ys,
                                            fsigx0(r),
                                            fsigx1(r),
                                            fsigy(r))
                ipx, ipy = np.meshgrid(px, py, indexing = 'ij')
                id_ele = wei > 1e-6
                nb_ele = np.sum(id_ele)
                row = [irow] * nb_ele
                col = ipx[id_ele] * image.shape[1] + ipy[id_ele]
                data = wei[id_ele]
                out[col, row] = data
        return out.tocoo()

    def z_main(self, image: Image):
        fsigz = interpolate.interp1d(self.fitted_z.uz, self.fitted_z.sigz,
                                     fill_value = 'extrapolate')
        out = sparse.lil_matrix((image.shape[2], image.shape[2]), dtype = np.float32)

        for i in nef.utils.tqdm(range(image.shape[2])):
            z = (i + 0.5) * image.unit_size[2] - image.size[2] / 2
            zmesh = (np.arange(image.shape[2]) - i) * image.unit_size[2]
            local_kernel = self.locate_kernel_z(zmesh, fsigz(abs(z)))
            spa_kernel = sparse.coo_matrix(local_kernel)
            id_ele = spa_kernel.data > 1e-16
            nb_ele = np.sum(id_ele)
            if nb_ele > 0:
                row = [i] * nb_ele
                col = spa_kernel.col[id_ele]
                data = spa_kernel.data[id_ele]
                out[col, row] = data
        return out.tocoo()

    def locate_kernel_z(self, mesh, sig):
        ans = 1.0 / np.sqrt(2 * np.pi * sig ** 2) * np.exp(-mesh ** 2 / 2 / sig ** 2)
        return ans / np.sum(ans)
