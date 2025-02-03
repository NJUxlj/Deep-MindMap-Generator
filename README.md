# Deep-MindMap-Generator
it's unfinished




## Project Structure
```Plain Text
dmg/  
├── configs/                  # 配置文件  
│   ├── model/  
│   │   └── llama2.yaml      # 模型超参数配置  
│   └── train.yaml           # 训练配置  
├── data/                    # 数据处理  
│   ├── loaders/             # 自定义数据加载器  
│   │   └── mindmap_loader.py  
│   └── processors/          # 数据预处理  
│       └── tree_processor.py  
├── src/  
│   ├── modeling/            # 核心模型架构  
│   │   ├── adapter.py       # LoRA适配层  
│   │   └── tree_decoder.py  # 树状结构解码器  
│   ├── evaluation/          # 评估指标（参考IEEE评估标准[^4][^5]）  
│   │   ├── structural_similarity.py  
│   │   └── semantic_accuracy.py  
│   └── utils/               # 工具函数  
│       └── visualization.py # 思维导图可视化  
├── scripts/                 # 实用脚本  
│   ├── train_ppo.py         # RLHF训练脚本  
│   └── export_model.py      # 模型导出  
├── tests/                   # 单元测试  
└── requirements.txt         # 依赖库  


```