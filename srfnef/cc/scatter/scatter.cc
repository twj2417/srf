#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"
#include <ctime>

using namespace tensorflow;

REGISTER_OP("ScatterOverLors")
	.Input("lors: float")
	.Input("ind1: int32")
	.Input("ind2: int32")
	.Input("umap_project: float")
	.Input("image_project: float")
	.Input("umap: float")
	.Input("grid: int32")
	.Input("center: float")
	.Input("size: float")
	.Input("smp: int32")
	.Output("projection_value: float")
	.Attr("low_eng: float")
	.Attr("high_eng: float")
	.Attr("res_eng: float")
	.Attr("angle_per_block: float")
	.Attr("crystal_area: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->Vector(c->Dim(c->input(0), 1)));
		// c->set_output(0, c->input(5));

		return Status::OK();
	});


REGISTER_OP("ScaleOverLors")
	.Input("lors: float")
	.Output("scale_value: float")
	.Attr("angle_per_block: float")
	.Attr("crystal_area: float")
	.Attr("epsilon_ab: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->Vector(c->Dim(c->input(0), 1)));
		return Status::OK();
	});

void scatter_loop_all_lors(const float *x1, const float *y1, const float *z1,
                           const float *x2, const float *y2, const float *z2,
                           const int *ind1, const int *ind2,
                           const float *umap_project, const float *image_project, const float *umap,
                           const int *grid, const float *center, const float *size, const int *smp,
                           const float low_eng, const float high_eng, const float res_eng,
                           const float angle_per_block, const float crystal_area,
                           const int num_events,
                           float *vproj);

void scale_loop_all_lors(const float *x1, const float *y1, const float *z1,
                         const float *x2, const float *y2, const float *z2,
                         const float angle_per_block, const float crystal_area,
                         const float epsilon_ab,
                         const int num_events,
                         float *lor_scales);

class ScatterOverLors : public OpKernel
{
public:
	explicit ScatterOverLors(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("low_eng", &low_eng));
		OP_REQUIRES_OK(context, context->GetAttr("high_eng", &high_eng));
		OP_REQUIRES_OK(context, context->GetAttr("res_eng", &res_eng));
		OP_REQUIRES_OK(context, context->GetAttr("angle_per_block", &angle_per_block));
		OP_REQUIRES_OK(context, context->GetAttr("crystal_area", &crystal_area));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor
		const Tensor &lors = context->input(0);
		const Tensor &ind1 = context->input(1);
		const Tensor &ind2 = context->input(2);
		const Tensor &umap_project = context->input(3);
		const Tensor &image_project = context->input(4);
		const Tensor &umap = context->input(5);

		const Tensor &grid = context->input(6);
		const Tensor &center = context->input(7);
		const Tensor &size = context->input(8);
		const Tensor &smp = context->input(9);

		// Create an output tensor
		Tensor *projection_value = NULL;

		// define the shape of output tensors.
		TensorShape out_shape;
		out_shape.AddDim(lors.shape().dim_size(1));

		OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
														 &projection_value));
		auto pv_flat = projection_value->flat<float>();

		cudaMemset(pv_flat.data(), 0, sizeof(float) * pv_flat.size());

		auto x1t = lors.Slice(0, 1);
		auto y1t = lors.Slice(1, 2);
		auto z1t = lors.Slice(2, 3);
		auto x2t = lors.Slice(3, 4);
		auto y2t = lors.Slice(4, 5);
		auto z2t = lors.Slice(5, 6);
		// std::cout<<"TEST1"<<std::endl;
		auto x1 = x1t.unaligned_flat<float>();
		auto y1 = y1t.unaligned_flat<float>();
		auto z1 = z1t.unaligned_flat<float>();
		auto x2 = x2t.unaligned_flat<float>();
		auto y2 = y2t.unaligned_flat<float>();
		auto z2 = z2t.unaligned_flat<float>();

		auto grid_flat = grid.flat<int>();
		auto center_flat = center.flat<float>();
		auto size_flat = size.flat<float>();
        auto smp_flat = smp.flat<int>();
		auto ind1_flat = ind1.flat<int>();
		auto ind2_flat = ind2.flat<int>();
		auto umap_project_flat = umap_project.flat<float>();
		auto image_project_flat = image_project.flat<float>();
		auto umap_flat = umap.flat<float>();
		unsigned int num_events = pv_flat.size();

		scatter_loop_all_lors(x1.data(), y1.data(), z1.data(),
							  x2.data(), y2.data(), z2.data(),
							  ind1_flat.data(), ind2_flat.data(),
							  umap_project_flat.data(), image_project_flat.data(), umap_flat.data(),
							  grid_flat.data(), center_flat.data(), size_flat.data(), smp_flat.data(),
							  low_eng, high_eng, res_eng,
							  angle_per_block, crystal_area, num_events,
							  pv_flat.data());
	}

private:
	float low_eng;
	float high_eng;
	float res_eng;
	float angle_per_block;
	float crystal_area;
};



class ScaleOverLorsOp : public OpKernel
{
public:
	explicit ScaleOverLorsOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("angle_per_block", &angle_per_block));
		OP_REQUIRES_OK(context, context->GetAttr("crystal_area", &crystal_area));
		OP_REQUIRES_OK(context, context->GetAttr("epsilon_ab", &epsilon_ab));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor
		const Tensor &lors = context->input(0);

		// Create an output tensor
		Tensor *scale_value = NULL;

		// define the shape of output tensors.
		TensorShape out_shape;
		out_shape.AddDim(lors.shape().dim_size(1));

		OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
														 &scale_value));
		auto sv_flat = scale_value->flat<float>();

		cudaMemset(sv_flat.data(), 0, sizeof(float) * sv_flat.size());

		auto x1t = lors.Slice(0, 1);
		auto y1t = lors.Slice(1, 2);
		auto z1t = lors.Slice(2, 3);
		auto x2t = lors.Slice(3, 4);
		auto y2t = lors.Slice(4, 5);
		auto z2t = lors.Slice(5, 6);
		// std::cout<<"TEST1"<<std::endl;
		auto x1 = x1t.unaligned_flat<float>();
		auto y1 = y1t.unaligned_flat<float>();
		auto z1 = z1t.unaligned_flat<float>();
		auto x2 = x2t.unaligned_flat<float>();
		auto y2 = y2t.unaligned_flat<float>();
		auto z2 = z2t.unaligned_flat<float>();

		unsigned int num_events = sv_flat.size();

		scale_loop_all_lors(x1.data(), y1.data(), z1.data(),
							x2.data(), y2.data(), z2.data(),
							angle_per_block, crystal_area, epsilon_ab,
							num_events,
							sv_flat.data());
	}


private:
	float angle_per_block;
	float crystal_area;
	float epsilon_ab;
};


#define REGISTER_GPU_KERNEL(name, op) \
	REGISTER_KERNEL_BUILDER(          \
		Name(name).Device(DEVICE_GPU), op)

REGISTER_GPU_KERNEL("ScatterOverLors", ScatterOverLors);
REGISTER_GPU_KERNEL("ScaleOverLors", ScaleOverLorsOp);

#undef REGISTER_GPU_KERNEL
