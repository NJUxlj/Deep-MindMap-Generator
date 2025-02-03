import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from transformers import LlamaForCausalLM
import torch.nn as nn

from typing import List, Dict, Tuple, Union

class DMGAdapter:
    def __init__(self, base_model_name="meta-llama/Llama-2-7b-hf"):
        # 初始化基础模型（参考DeepSeek-R1架构）
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_4bit=True,  # 使用bitsandbytes量化
            device_map="auto"
        )
        
        # 配置LoRA参数（参数高效微调）
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # 应用PEFT适配器
        self.model = get_peft_model(self.base_model, self.lora_config)
        self.model.print_trainable_parameters()

    def generate_mindmap(self, input_text, max_depth=5):
        """
        生成树状思维导图的核心方法
        参数：
            input_text: 输入的主题文本
            max_depth: 最大展开层级（参考DeepMind Fast Mapping[^2]）
        返回：
            tree_structure: 树状结构字典
        """
        # 将输入文本编码为层次化表示
        encoded_input = self._encode_hierarchical(input_text)
        
        # 使用改进的树状解码算法
        return self.tree_decoder.decode(
            encoded_input,
            max_depth=max_depth,
            branching_factor=3
        )       





class HierarchicalLoRA(nn.Module):
    def __init__(self, base_model: LlamaForCausalLM, lora_config: LoraConfig):
        super().__init__()
        self.base_model = base_model
        self.lora_config = lora_config
        
        # 应用LoRA到关键层（参考DeepSeek-R1的适配策略）
        self._apply_lora_to_layers([
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.12.self_attn.q_proj"  # 跨层适配
        ])
        
    def _apply_lora_to_layers(self, target_modules: List[str]):
        """动态注入LoRA层"""
        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.lora_config.lora_dropout,
            bias="none",
            modules_to_save=["lm_head"]  # 保留输出层可训练
        )
        self.model = get_peft_model(self.base_model, peft_config)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )