#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"
#include <ctime>

using namespace tensorflow;

REGISTER_OP("ProjectionTof")
	.Input("lors: float")
	.Input("image: float")
	.Input("grid: int32")
	.Input("center: float")
	.Input("size: float")
	.Input("tof_values: float")
	.Output("line_integral: float")
	.Attr("tof_bin: float")
	.Attr("tof_sigma2: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		c->set_output(0, c->Vector(c->Dim(c->input(0), 1)));
		return Status::OK();
	});

REGISTER_OP("BackprojectionTof")
	.Input("image: float")
	.Input("lors: float")
	.Input("lors_value: float")
	.Input("grid: int32")
	.Input("center: float")
	.Input("size: float")
    .Input("tof_values: float")
	.Output("backpro_image: float")
	.Attr("tof_bin: float")
	.Attr("tof_sigma2: float")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
		//set the size of backpro_image the same as the input image.
		c->set_output(0, c->input(0));
		return Status::OK();
	});
void projection(const float *x1, const float *y1, const float *z1,
				const float *x2, const float *y2, const float *z2,
				float *projection_value,
				const int *grid, const float *center, const float *size,
				const float* tof_values, const float tof_bin, const float tof_sigma2,
				const float *image, const int num_events);

void backprojection(const float *x1, const float *y1, const float *z1,
					const float *x2, const float *y2, const float *z2,
					const float *projection_value,
					const int *grid, const float *center, const float *size,
                    const float* tof_values, const float tof_bin, const float tof_sigma2,
					float *image, const int num_events);

class ProjectionTofOp : public OpKernel
{
public:
	explicit ProjectionTofOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("tof_bin", &tof_bin));
		OP_REQUIRES_OK(context, context->GetAttr("tof_sigma2", &tof_sigma2));
	}

	void Compute(OpKernelContext *context) override
	{
		// Grab the input tensor
		const Tensor &lors = context->input(0);
		const Tensor &image = context->input(1);
		const Tensor &grid = context->input(2);
		const Tensor &center = context->input(3);
		const Tensor &size = context->input(4);
		const Tensor &tof_values = context->input(5);
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
		auto image_flat = image.flat<float>();
		unsigned int num_events = pv_flat.size();
		auto tof_flat = tof_values.flat<float>();


		projection(x1.data(), y1.data(), z1.data(),
				       x2.data(), y2.data(), z2.data(),
				       pv_flat.data(),
				       grid_flat.data(), center_flat.data(), size_flat.data(),
				       tof_flat.data(), tof_bin, tof_sigma2,
				       image_flat.data(), num_events);
	}

private:
	float tof_bin;
	float tof_sigma2;
};

class BackprojectionTofOp : public OpKernel
{
public:
	explicit BackprojectionTofOp(OpKernelConstruction *context) : OpKernel(context)
	{
        OP_REQUIRES_OK(context, context->GetAttr("tof_bin", &tof_bin));
		OP_REQUIRES_OK(context, context->GetAttr("tof_sigma2", &tof_sigma2));
	}

	void Compute(OpKernelContext *context) override
	{

		// Grab the geometries of an image.
		const Tensor &image = context->input(0);
		const Tensor &lors = context->input(1);
		//grab the input projection values(line integral)
		const Tensor &projection_value = context->input(2);
		const Tensor &grid = context->input(3);
		const Tensor &center = context->input(4);
		const Tensor &size = context->input(5);
        const Tensor &tof_values = context->input(6);
		// Create an output backprojected image
		Tensor *backpro_image = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, image.shape(),
														 &backpro_image));
		//set the initial backprojection image value to zero.
		auto backpro_image_flat = backpro_image->flat<float>();
        cudaMemset(backpro_image_flat.data(), 0, sizeof(float) * backpro_image_flat.size());

		// auto backpro_image_flat = backpro_image->flat<float>();

		auto pv_flat = projection_value.flat<float>();

		auto x1t = lors.Slice(0, 1);
		auto y1t = lors.Slice(1, 2);
		auto z1t = lors.Slice(2, 3);
		auto x2t = lors.Slice(3, 4);
		auto y2t = lors.Slice(4, 5);
		auto z2t = lors.Slice(5, 6);

		auto x1 = x1t.unaligned_flat<float>();
		auto y1 = y1t.unaligned_flat<float>();
		auto z1 = z1t.unaligned_flat<float>();
		auto x2 = x2t.unaligned_flat<float>();
		auto y2 = y2t.unaligned_flat<float>();
		auto z2 = z2t.unaligned_flat<float>();

		auto grid_flat = grid.flat<int>();
		auto center_flat = center.flat<float>();
		auto size_flat = size.flat<float>();
		unsigned int num_events = pv_flat.size();
		auto tof_flat = tof_values.flat<float>();

		// cudaDeviceSynchronize();
		// std::cout << "BackProjection pre kernel time cost:" << clock() - t << std::endl;
		backprojection(x1.data(), y1.data(), z1.data(),
					       x2.data(), y2.data(), z2.data(),
					       pv_flat.data(),
					       grid_flat.data(), center_flat.data(), size_flat.data(),
					       tof_flat.data(), tof_bin, tof_sigma2,
					       backpro_image_flat.data(), num_events);
	}

private:
	// string model;
	float tof_bin;
	float tof_sigma2;
};

#define REGISTER_GPU_KERNEL(name, op) \
	REGISTER_KERNEL_BUILDER(          \
		Name(name).Device(DEVICE_GPU), op)

REGISTER_GPU_KERNEL("ProjectionTof", ProjectionTofOp);
REGISTER_GPU_KERNEL("BackprojectionTof", BackprojectionTofOp);

#undef REGISTER_GPU_KERNEL
