import numpy as np
from sklearn.metrics import f1_score

class StructuralEvaluator:
    def __init__(self, reference_maps):
        """
        参数：
            reference_maps: 参考思维导图数据集（需符合MindBench格式[^6]）
        """
        self.reference = reference_maps
    
    def calculate_similarity(self, generated_map):
        """
        计算生成结果与参考集的结构相似性
        实现指标：
        - 节点层级一致性 (NLC)
        - 关系准确性 (RA)
        - 结构完整性指数 (SII)
        """
        # 提取结构特征
        gen_features = self._extract_structural_features(generated_map)
        ref_features = [self._extract_structural_features(m) for m in self.reference]
        
        # 计算相似性得分
        scores = {
            'NLC': self._calc_level_consistency(gen_features, ref_features),
            'RA': self._calc_relation_accuracy(gen_features, ref_features),
            'SII': self._calc_structural_integrity(gen_features)
        }
        return scores

    def _extract_structural_features(self, mindmap):
        """提取层级结构特征"""
        return {
            'depth': mindmap.max_depth,
            'branching_factors': mindmap.branching_stats,
            'node_relations': mindmap.relation_edges
        }