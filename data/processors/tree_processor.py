from transformers import PreTrainedTokenizer
import torch

class TreeProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 添加特殊token
        tokenizer.add_special_tokens({"additional_special_tokens": self.special_tokens})
        
    def tokenize_tree(self, tree_seq: str) -> Dict:
        """将序列化树结构转换为模型输入"""
        encoding = self.tokenizer(
            tree_seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": encoding["input_ids"].clone()  # 用于自回归训练
        }
    
    def decode_to_tree(self, logits: torch.Tensor) -> Dict:
        """将模型输出解码回树结构"""
        tokens = logits.argmax(dim=-1)[0]
        decoded = self.tokenizer.decode(tokens, skip_special_tokens=False)
        return self._parse_sequence(decoded)
    
    def _parse_sequence(self, seq: str) -> Dict:
        """解析标记序列到树结构（使用栈式解析）"""
        stack = []
        current_node = {}
        for token in seq.split():
            if token == "<node>":
                new_node = {"text": "", "children": []}
                if stack:
                    stack[-1]["children"].append(new_node)
                stack.append(new_node)
            elif token == "</node>":
                current_node = stack.pop()
            elif token == "<child>":
                pass  # 层级进入
            elif token == "</child>":
                pass  # 层级返回
            else:
                if stack:
                    stack[-1]["text"] += " " + token
        return current_node