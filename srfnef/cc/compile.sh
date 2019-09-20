# !/bin/sh
rm -rf BUILD siddon siddon_tof deform
cp $NEF_USER_OP_SOURCE/BUILD .
cp -r $NEF_USER_OP_SOURCE/siddon .
cp -r $NEF_USER_OP_SOURCE/siddon_tof .
cp -r $NEF_USER_OP_SOURCE/deform .
# cp -rf $NEF_USER_OP_SOURCE/scatter .

bazel build -c opt //tensorflow/core/user_ops:tf_siddon_module.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
bazel build -c opt //tensorflow/core/user_ops:tf_siddon_tof_module.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
bazel build -c opt //tensorflow/core/user_ops:tf_deform_module.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
bazel build -c opt //tensorflow/core/user_ops:tf_deform_tex_module.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
# bazel build -c opt //tensorflow/core/user_ops:tf_scatter_module.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --verbose_failures

bazel build -c opt //tensorflow/core/user_ops:tf_siddon_2d_tof.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"