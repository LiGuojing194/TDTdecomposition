from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_cuda_extension',  # 可以修改为更通用的名称，因为现在包含多个扩展
    ext_modules=[
        CUDAExtension(
            name='mycudaf',  # 保持原有扩展模块
            sources=[
            #    '/root/autodl-tmp/TCRTruss32/src/demo_truss/myops/mysrc/cuda_extension/segment_add.cpp',
            #     '/root/autodl-tmp/TCRTruss32/src/demo_truss/myops/mysrc/cuda_extension/segment_add_kernel.cu'
            '/root/autodl-tmp/TCRTruss32/src/demo_truss/myops/mysrc/cuda_extension/mycuda_extension.cpp',
            '/root/autodl-tmp/TCRTruss32/src/demo_truss/myops/mysrc/cuda_extension/mycuda_extension_kernel.cu'
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
