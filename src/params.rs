use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::{Dtype, SafeTensors};
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| -> Tensor<f32> {
            // 根据名称获取 tensor
            let tensor_view = safetensor
                .tensor(name)
                .expect(&format!("Tensor {} not found", name));

            assert_eq!(
                tensor_view.dtype(),
                Dtype::F32,
                "Expected tensor {} to have dtype F32, but found {:?}",
                name,
                tensor_view.dtype()
            );

            // 将原始字节转换为 f32 类型并创建 Tensor
            let data = tensor_view
                .data()
                .chunks(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();

            Tensor::new(data, &tensor_view.shape().to_vec())
        };

        let n_layers = config.num_hidden_layers;

        let get_layer_tensors = |prefix: &str| -> Vec<Tensor<f32>> {
            (0..n_layers)
                .map(|layer_idx| get_tensor(&format!("model.layers.{layer_idx}.{}", prefix)))
                .collect()
        };

        LLamaParams {
            embedding_table: if config.tie_word_embeddings {
                get_tensor("lm_head.weight")
            } else {
                get_tensor("model.embed_tokens.weight")
            },
            rms_att_w: get_layer_tensors("input_layernorm.weight"),
            wq: get_layer_tensors("self_attn.q_proj.weight"),
            wk: get_layer_tensors("self_attn.k_proj.weight"),
            wv: get_layer_tensors("self_attn.v_proj.weight"),
            wo: get_layer_tensors("self_attn.o_proj.weight"),
            rms_ffn_w: get_layer_tensors("post_attention_layernorm.weight"),
            w_up: get_layer_tensors("mlp.up_proj.weight"),
            w_gate: get_layer_tensors("mlp.gate_proj.weight"),
            w_down: get_layer_tensors("mlp.down_proj.weight"),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
