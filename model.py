import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertModule(nn.Module):
    """专家网络模块"""

    def __init__(self, input_dim, hidden_dim):
        super(ExpertModule, self).__init__()
        self.expert = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.expert(x)


class GateModule(nn.Module):
    """门控网络模块"""

    def __init__(self, input_dim, num_experts):
        super(GateModule, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.gate(x)


class PLELayer(nn.Module):
    """PLE层"""

    def __init__(self, input_dim, hidden_dim, num_experts_per_task, num_shared_experts, num_tasks=5):
        super(PLELayer, self).__init__()
        self.num_tasks = num_tasks
        self.num_experts_per_task = num_experts_per_task
        self.num_shared_experts = num_shared_experts

        # 共享专家
        self.shared_experts = nn.ModuleList([
            ExpertModule(input_dim, hidden_dim)
            for _ in range(num_shared_experts)
        ])

        # 任务特定专家
        self.task_specific_experts = nn.ModuleList([
            nn.ModuleList([
                ExpertModule(input_dim, hidden_dim)
                for _ in range(num_experts_per_task)
            ])
            for _ in range(num_tasks)
        ])

        # 任务门控
        self.gates = nn.ModuleList([
            GateModule(input_dim, num_experts_per_task + num_shared_experts)
            for _ in range(num_tasks)
        ])

    def forward(self, x):
        # 计算所有专家的输出
        shared_expert_outputs = [expert(x) for expert in self.shared_experts]

        task_outputs = []
        for task_id in range(self.num_tasks):
            # 获取任务特定专家的输出
            task_expert_outputs = [
                expert(x) for expert in self.task_specific_experts[task_id]
            ]

            # 组合所有相关专家的输出
            combined_experts = task_expert_outputs + shared_expert_outputs

            # 计算门控权重
            gate_weights = self.gates[task_id](x)

            # 加权组合专家输出
            task_output = torch.zeros_like(combined_experts[0])
            for i, expert_output in enumerate(combined_experts):
                task_output += gate_weights[:, i].unsqueeze(1) * expert_output

            task_outputs.append(task_output)

        return task_outputs


class AntibacterialPeptidePredictor(nn.Module):
    """抗菌肽活性预测模型"""

    def __init__(self, input_dim=40, hidden_dims=[128, 64],
                 num_experts_per_task=3, num_shared_experts=2):
        super(AntibacterialPeptidePredictor, self).__init__()

        self.num_tasks = 5  # 五种菌群

        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.3)
        )

        # PLE层
        self.ple_layers = nn.ModuleList([
            PLELayer(
                input_dim=hidden_dims[0] if i == 0 else hidden_dims[1],
                hidden_dim=hidden_dims[1],
                num_experts_per_task=num_experts_per_task,
                num_shared_experts=num_shared_experts,
                num_tasks=self.num_tasks
            )
            for i in range(2)  # 使用2个PLE层
        ])

        # 任务特定输出层
        self.task_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[1], 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(0.3),
                nn.Linear(32, 1),
                nn.Sigmoid()  # 输出活性预测概率
            )
            for _ in range(self.num_tasks)
        ])

    def forward(self, x):
        # 特征提取
        x = self.feature_extractor(x)

        # 通过PLE层
        task_features = x
        for ple_layer in self.ple_layers:
            task_features = ple_layer(task_features)

        # 任务特定预测
        outputs = []
        for task_id in range(self.num_tasks):
            task_output = self.task_towers[task_id](task_features[task_id])
            outputs.append(task_output)

        return outputs