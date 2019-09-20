load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
   name = "tf_siddon_module.so",
   srcs = ["siddon/siddon.cc"],
   gpu_srcs = ["siddon/siddon.cu.cc"]
)

tf_custom_op_library(
   name = "tf_siddon_tof_module.so",
   srcs = ["siddon_tof/siddon_tof.cc"],
   gpu_srcs = ["siddon_tof/siddon_tof.cu.cc"]
)

tf_custom_op_library(
   name = "tf_deform_module.so",
   srcs = ["deform/deform.cc"],
   gpu_srcs = ["deform/deform.cu.cc"]
)

tf_custom_op_library(
   name = "tf_deform_tex_module.so",
   srcs = ["deform/deform_tex.cc"],
   gpu_srcs = ["deform/deform_tex.cu.cc"]
)
