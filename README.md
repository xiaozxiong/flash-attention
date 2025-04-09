# Dive into FlashAttention
## FLA APIs

```cpp
//TODO: bind functions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    m.def("fwd", &FLASH_NAMESPACE::mha_fwd, "Forward pass");
    m.def("varlen_fwd", &FLASH_NAMESPACE::mha_varlen_fwd, "Forward pass (variable length)");
    m.def("bwd", &FLASH_NAMESPACE::mha_bwd, "Backward pass");
    m.def("varlen_bwd", &FLASH_NAMESPACE::mha_varlen_bwd, "Backward pass (variable length)");
    m.def("fwd_kvcache", &FLASH_NAMESPACE::mha_fwd_kvcache, "Forward pass, with KV-cache");
}
```

Currently, FLA can compute attention with paged KV cache, but in this case variable length is not supported. When we want varible length feature, we can not have PagedAttention. Besides, FLA doesn't support custom mask except causal mask.

## Flash-Decoding for long-context inference

Due to the fairly short q sequence and much longer kv sequences, spliting q would lead to low utilization of GPU. Therefore, we need to split k and v, then recuce the result across SMs. This strategy is different with workload partition in traing and prefill.

## CUDA Version

### Core files

- `csrc/flash_attn/flash_api.cpp`
- `csrc/flash_attn/src/flash.h`
- `csrc/flash_attn/src/flash_fwd_launch_template.h`
- `csrc/flash_attn/src/flash_fwd_kernel.h`

### `flash.h`

Define parameter struct used in flash computation:
- `Qkv_params`: paramters about Q, K, V matrices
- `Flash_fwd_params`: inherits from `Qkv_params` and defines variables for forward pass and inference, paged attention, 
- `Flash_bwd_params`:

Three function templates declaration:
```cpp
template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, int Headdim, bool Is_causal> void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream);
```
Definiton of `run_mha_fwd_splitkv_dispatch()` locates `flash_fwd_launch_template.h`.
Definitions of `run_mha_fwd_()` and `run_mha_bwd_()` are generated via `generate_kernels.py`.

### `generate_kernels.py`

Generate content of .cu files such as `flash_fwd_split_hdim32_bf16_sm80.cu`, all files iwith split contain different **template instantiations** of `run_mha_fwd_splitkv_dispatch`.

Also generate **template specialization** of `run_mha_fwd_` and `run_mha_bwd_` for specific parameters in `flash_fwd_hdim192_fp16_sm80.cu` format file. So the definition of function should be included.

It seems that I can write a function template and use template specialization directly without a generic function template definition. Because I can not find the generic function templates of `run_mha_fwd_` and `run_mha_bwd_`.

```python
template<>
void run_mha_bwd_<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}>(Flash_bwd_params &params, cudaStream_t stream) {{
    run_mha_bwd_hdim{HEAD_DIM}<{DTYPE}, {IS_CAUSAL}>(params, stream);
}}

template<>
void run_mha_fwd_<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}>(Flash_fwd_params &params, cudaStream_t stream) {{
    run_mha_fwd_hdim{HEAD_DIM}<{DTYPE}, {IS_CAUSAL}>(params, stream);
}}
```
### `flash_fwd_launch_template.h`

Define templates which are included in other files like `flash_fwd_split_hdim32_bf16_sm80.cu` in which specific template parameters are assigned.

For inference, I would like to focus on `run_mha_fwd_splitkv_dispatch`.

Kernels, e.g., `flash_fwd_splitkv_kernel`, are defined through macro and generated during preprocessing. 

### `flash_fwd_kernel.h`
Device functions used in kernels defined in `flash_fwd_launch_template.h`.

## Triton Version

### Custom Mask

[flashattention2-custom-mask](https://github.com/alexzhang13/flashattention2-custom-mask.git) is a triton version of FLA2 which supports custom mask.

## Similar Works

### FlashInfer


### FlexAttention in PyTorch