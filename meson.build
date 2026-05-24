project(
  'hand_eval_simd',
  'cpp',
  version : '0.1',
  meson_version : '>= 1.3.0',
  default_options : ['warning_level=3', 'cpp_std=c++23'],
)

simde_proj = subproject('simde')
simde_dep = simde_proj.get_variable('simde_dep')

omp = dependency('openmp')

dependencies = [
  simde_dep,
  omp
]

cpp = meson.get_compiler('cpp')
opt_args = cpp.get_supported_arguments(['-march=native', '-O3', '-openmp'])

exe = executable(
  'hand_eval_simd',
  'hand_eval_simd.cpp',
  install : true,
  dependencies : dependencies,
  cpp_args : opt_args
)

test('basic', exe)
