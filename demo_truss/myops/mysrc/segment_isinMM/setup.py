from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='isinmm_extension',  # 可以修改为更通用的名称，因为现在包含多个扩展
    ext_modules=[
        CUDAExtension(
            name='myisinmm',  # 保持原有扩展模块
            sources=[
            '/root/autodl-tmp/TCRTruss32/src/demo_truss/myops/mysrc/segment_isinMM/segment_isinMM.cpp',
            '/root/autodl-tmp/TCRTruss32/src/demo_truss/myops/mysrc/segment_isinMM/segment_isinMM_kernel.cu'
            ],
            extra_compile_args={
                'cxx': ['-std=c++17', '-O3'],
                'nvcc': [
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__',
                    '--expt-relaxed-constexpr',
                    '-gencode=arch=compute_70,code=compute_70',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-std=c++17',
                    '-O3'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
