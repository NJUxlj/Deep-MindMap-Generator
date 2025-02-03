from datasets import Dataset
from xml.etree import ElementTree as ET
from typing import List, Dict

class MindMapLoader:
    def __init__(self, max_depth=6, branching_factor=4):
        """
        加载FreeMind格式的思维导图数据
        参数：
            max_depth: 最大处理深度（参考MindBench数据集规范）
            branching_factor: 分支因子限制
        """
        self.special_tokens = ["<node>", "</node>", "<child>", "</child>"]
        
    def parse_freemind(self, xml_path: str) -> Dict:
        """解析FreeMind XML文件为树结构"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return self._parse_node(root[0])  # 第一个节点为根
    
    def _parse_node(self, node, depth=0) -> Dict:
        """递归解析节点"""
        node_data = {
            "text": node.attrib["TEXT"],
            "depth": depth,
            "children": []
        }
        for child in node.findall("node"):
            node_data["children"].append(self._parse_node(child, depth+1))
        return node_data

    def convert_to_sequence(self, tree: Dict) -> str:
        """
        将树结构转换为训练序列（使用类似XML的标记）
        示例：
        <node>Root<child><node>Child1</node><node>Child2</node></child></node>
        """
        seq = f"<node>{tree['text']}"
        if tree["children"]:
            seq += "<child>"
            for child in tree["children"]:
                seq += self.convert_to_sequence(child)
            seq += "</child>"
        return seq + "</node>"