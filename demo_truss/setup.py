from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_extensions',  # 可以修改为更通用的名称，因为现在包含多个扩展
    ext_modules=[
        CUDAExtension(
            name='segment_add_extension',  # 保持原有扩展模块
            sources=[
                '/root/autodl-tmp/TCRTruss32/src/demo_truss/myops/mysrc/run_segment_add/segment_add.cpp',
                '/root/autodl-tmp/TCRTruss32/src/demo_truss/myops/mysrc/run_segment_add/segment_add_kernel.cu'
            ]
        ),
        CUDAExtension(
            name='segment_isin_extension',  # 新的扩展模块
            sources=[
                '/root/autodl-tmp/TCRTruss32/src/demo_truss/myops/mysrc/run_segment_isin/segment_isin.cpp',  # 指向新的 .cpp 文件的路径
                '/root/autodl-tmp/TCRTruss32/src/demo_truss/myops/mysrc/run_segment_isin/segment_isin_kernel.cu'  # 指向新的 .cu 文件的路径
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
