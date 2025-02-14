mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::io::{self, Write};
use std::path::PathBuf;
use tokenizers::Tokenizer;

struct Message {
    role: String,
    msg: String,
}

impl Message {
    fn format(&self) -> String {
        format!("<|im_start|>{}\n{}<|im_end|>\n", self.role, self.msg)
    }
}

fn main() {
    let mode = "chat"; // "story"、"chat"

    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join(mode);
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    if mode == "chat" {
        chat(&llama, &tokenizer, 1.0);
    } else if mode == "story" {
        let input = "Once upon a time";
        let binding = tokenizer.encode(input, true).unwrap();
        let input_ids = binding.get_ids();
        let output_ids = llama.generate(input_ids, 500, 0.8, 30, 1.0);
        println!("{}", tokenizer.decode(&output_ids, true).unwrap());
    }
}
fn chat(llama: &model::Llama<f32>, tokenizer: &Tokenizer, temperature: f32) {
    let mut kvcache = llama.new_cache();
    let mut conversation_history: Vec<Message> = vec![]; //存储Message结构对话消息
    let mut formatted_input = String::new(); // 存储经过Jinja2模板格式化后的对话输入

    loop {
        print!("User: \n");
        io::stdout().flush().unwrap();

        // 读取用户输入
        let user_input = {
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap(); // 读取用户输入
            input.trim().to_string() // 去掉末尾的换行符并返回
        };

        // 将用户消息添加到对话历史中
        conversation_history.push(Message {
            role: "user".to_string(),
            msg: user_input.to_string(),
        });

        // 将最新的用户消息格式化并追加到 formatted_input 中
        let formatted_user_input = conversation_history.last().unwrap().format(); // 格式化最新的用户消息
        formatted_input.push_str(&formatted_user_input); // 将格式化的用户消息追加到对话输入中

        // 使用 tokenizer 将对话输入编码成 token ids
        let encoded = tokenizer
            .encode(formatted_input.clone() + "<|im_start|>assistant\n", true)
            .unwrap();
        let input_ids = encoded.get_ids();
        let mut generated_tokens = vec![];

        // 开始生成模型的回答
        println!("Assistant: ");
        io::stdout().flush().unwrap();

        // 调用模型的 stream_generate 方法生成模型的回答
        let response_tokens =
            llama.stream_generate(input_ids, 500, 0.8, 30, temperature, &mut kvcache);
        for token in response_tokens {
            generated_tokens.push(token);
        }

        // 使用 tokenizer 将生成的 token 转换为文本
        let response_text = tokenizer.decode(&generated_tokens, true).unwrap();
        println!("{}", response_text);
        io::stdout().flush().unwrap();

        // 将模型的回答添加到对话历史中
        conversation_history.push(Message {
            role: "assistant".to_string(),
            msg: response_text,
        });
        // 将模型回答格式化后追加到 formatted_input 中
        let formatted_assistant_response = conversation_history.last().unwrap().format(); // Format the assistant response
        formatted_input.push_str(&formatted_assistant_response); // Append to the running conversation_input
    }
}
