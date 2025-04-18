```mermaid
flowchart LR
    subgraph StateEncoding[State Encoding]
        A1[手牌/公共牌<br>4×15 One-Hot] --> SE1
        A2[对局历史<br>(出牌序列等)] --> SE2[LSTM编码]
        A3[任务状态追踪<br>(consecutive_plays, last_player)] --> SE3
        SE1 --> state_embed
        SE2 --> state_embed
        SE3 --> state_embed
    end

    subgraph ActionEncoding[Action Encoding]
        B1[当前候选动作<br>4×15 One-Hot] --> AC1[MLP或Conv等]
        AC1 --> action_embed
    end

    state_embed --> Combine1
    action_embed --> Combine1
    subgraph Combine1[特征拼接/融合]
        direction TB
        CombineIn[(拼接State向量<br>+ Action向量)]
        CombineIn --> CombineOut
    end
    
    CombineOut --> MLP1[MLP × 6层<br>输出Q值]

    subgraph CooperationModule[农民协作模块(Reward Shaping)]
        C1[(task_state:<br>last_player,<br>hand_cards,<br>bomb_played...)]
        C2[局部协作奖励计算<br>(连续出牌, 压制等)]
        C3[终局协作奖励<br>(农民胜利,<br>出炸弹火箭等)]
        C1 --> C2
        C2 --> CoopR[协作奖励]
        C3 --> CoopR
    end

    MLP1 --> QValue[Q(s,a)]
    CoopR[协作奖励] -.合并到训练目标.-> QValue

    style CooperationModule fill:#CDEAFE,stroke:#999,stroke-width:1px
    style StateEncoding fill:#ffffcc,stroke:#999,stroke-width:1px
    style ActionEncoding fill:#ffffcc,stroke:#999,stroke-width:1px
    style Combine1 fill:#ECECEC,stroke:#999,stroke-width:1px
    style MLP1 fill:#FFEEE5,stroke:#999,stroke-width:1px
    style QValue fill:#FFFDD0,stroke:#999,stroke-width:1px
```