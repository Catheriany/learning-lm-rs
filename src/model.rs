use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );
            OP::matmul_transb(
                &mut residual,
                1.,
                &hidden_states,
                &self.params.wo[layer],
                1.,
            );
            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        let mut result_tokens = token_ids.to_vec();
        let mut kvcache = self.new_cache();
        let mut input_tensors: Tensor<u32> =
            Tensor::<u32>::new(result_tokens.clone(), &vec![result_tokens.len()]);

        // 生成 tokens 直到达到 max_len 或遇到 EOS (end of sentence)
        while result_tokens.len() < max_len {
            // 调用 forward 函数计算 logits
            let logits = self.forward(&input_tensors, &mut kvcache);

            // 调用random sample函数根据logits的值生成下一个 token
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);
            result_tokens.push(next_token);

            // 如果生成的 token 是 EOS (end of sentence)，则结束生成过程
            if next_token == self.eos_token_id {
                break;
            }
            // 更新输入张量，将新生成的 token 作为下一个输入
            input_tensors = Tensor::<u32>::new(vec![next_token], &vec![1]);
        }
        result_tokens
    }
    pub fn stream_generate<'a>(
        &'a self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        kvcache: &'a mut KVCache<f32>,
    ) -> impl Iterator<Item = u32> + 'a {
        let mut result_tokens = token_ids.to_vec();
        let mut input_tensors =
            Tensor::<u32>::new(result_tokens.clone(), &vec![result_tokens.len()]);

        std::iter::from_fn(move || {
            if result_tokens.len() >= max_len {
                return None;
            }

            let logits = self.forward(&input_tensors, kvcache);
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);
            result_tokens.push(next_token);
            input_tensors = Tensor::<u32>::new(vec![next_token], &vec![1]);

            if next_token == self.eos_token_id {
                None
            } else {
                Some(next_token) // 返回生成的 token
            }
        })
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();

    let head_dim = dqkv; // 每个头的维度
    let num_kv_heads = n_kv_h; // KV头数
    let query_heads_per_kv_group = n_groups; // 每组的查询头数

    let att_idx_3 = query_heads_per_kv_group * seq_len * total_seq_len;
    let att_idx_2 = seq_len * total_seq_len;
    let att_idx_1 = total_seq_len;

    for k_head in 0..num_kv_heads {
        let k_head_offset = k_head * head_dim;

        for q_group in 0..query_heads_per_kv_group {
            let att_head_idx = k_head * query_heads_per_kv_group + q_group;
            let q_head_offset = att_head_idx * head_dim;

            for seq_idx in 0..seq_len {
                let q_start_idx = seq_idx * num_kv_heads * n_groups * head_dim + q_head_offset;

                for total_seq_idx in 0..total_seq_len {
                    let k_start_idx = total_seq_idx * num_kv_heads * head_dim + k_head_offset;

                    let q_vec: &[f32] = &q_data[q_start_idx..q_start_idx + head_dim];
                    let k_vec = &k_data[k_start_idx..k_start_idx + head_dim];

                    // 计算注意力分数 (Q @ K^T / sqrt(d))
                    let score = q_vec.iter().zip(k_vec).map(|(&q, &k)| q * k).sum::<f32>()
                        / (head_dim as f32).sqrt();

                    // 计算得分在 att_scores 中的位置
                    let att_score_idx = k_head * att_idx_3
                        + q_group * att_idx_2
                        + seq_idx * att_idx_1
                        + total_seq_idx;

                    // 将计算出的注意力分数存入 `att_scores`
                    unsafe {
                        att_scores.data_mut()[att_score_idx] = score;
                    }
                }
            }
        }
    }

    // 对注意力分数进行掩码 softmax 归一化
    OP::masked_softmax(att_scores);

    let att_scores_data: &[f32] = att_scores.data();
    let hidden_states_data: &mut [f32] = unsafe { hidden_states.data_mut() };

    for v_head in 0..num_kv_heads {
        let v_head_offset: usize = v_head * head_dim;

        for q_group in 0..query_heads_per_kv_group {
            let att_head_offset = q_group * att_idx_2 + v_head * att_idx_3;

            for seq_idx in 0..seq_len {
                let att_vec_start_idx = seq_idx * att_idx_1 + att_head_offset;

                for head_dim_idx in 0..head_dim {
                    let v_start_idx = head_dim_idx + v_head_offset;
                    let mut weighted_sum = 0.0;

                    for total_seq_idx in 0..total_seq_len {
                        let att_idx = att_vec_start_idx + total_seq_idx;
                        let v_idx = v_start_idx + total_seq_idx * num_kv_heads * head_dim;

                        // 计算注意力加权值
                        weighted_sum += att_scores_data[att_idx] * v_data[v_idx];
                    }

                    // 计算 hidden_states 的索引，并存储计算结果
                    let hidden_idx = seq_idx * num_kv_heads * query_heads_per_kv_group * head_dim
                        + v_head * query_heads_per_kv_group * head_dim
                        + q_group * head_dim
                        + head_dim_idx;
                    hidden_states_data[hidden_idx] = weighted_sum;
                }
            }
        }
    }
}

fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    OP::rms_norm(hidden_states, &residual, &rms_w, eps);
    OP::matmul_transb(gate, 0., &hidden_states, &w_gate, 1.);
    OP::matmul_transb(up, 0., &hidden_states, &w_up, 1.);
    OP::swiglu(up, &gate);
    OP::matmul_transb(residual, 1., &up, &w_down, 1.);
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use crate::tensor::float_eq;
    use std::path::PathBuf;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(
        &model.params.embedding_table.data()[50],
        &0.14453125,
        1e-6
    ));
    assert_eq!(
        model.params.lm_head.data()[10],
        model.params.embedding_table.data()[10]
    );
    assert!(float_eq(
        &model.params.rms_att_w[0].data()[10],
        &0.18652344,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_ffn_w[1].data()[10],
        &0.32421875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_out_w.data()[100],
        &0.73046875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.w_down[0].data()[100],
        &-0.0625,
        1e-6
    ));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(
        &model.params.w_gate[1].data()[100],
        &0.296875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wq[1].data()[100],
        &0.032226563,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wk[1].data()[100],
        &-0.21386719,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wv[0].data()[100],
        &0.041015625,
        1e-6
    ));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));
}
