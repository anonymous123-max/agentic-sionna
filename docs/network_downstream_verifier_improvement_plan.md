# 3D 场景生成后的下游无线网络任务 Verifier 改进建议

## 1. 总体评价

当前方案方向是很好的：它把 3D 场景生成从“视觉/几何任务”提升成了“可执行的无线网络仿真基底”。

换句话说，scene 不只是生成出来看，而是必须能支撑以下下游任务：

- 覆盖率计算；
- AP placement / optimization；
- 材料与频率影响分析；
- 场景编辑后的重新计算；
- RT-to-PHY 链路级仿真；
- IRC 建筑规范与无线覆盖联合约束；
- 多小区系统级调度、公平性、吞吐量评估。

当前 C1–C7 已经覆盖了从几何有效性、RT 覆盖、PHY BER 到系统级公平性的完整链条。  
如果要让这个 benchmark 更稳、更像论文级方案，建议从 verifier 结构、artifact schema、数值一致性、baseline 对比和真实仿真模式几个方面进一步加强。

---

## 2. 建议把 verifier 分成三层

现在每个 capability 都包含 5 个 core checks，再加 capability-specific checks 和 oracle。这个结构是对的，但建议在文档和代码里明确分成三层。

### Layer A：Scene Validity

这一层只判断 3D 场景是否可作为网络仿真的输入。

建议包含：

- `scene_state.json` 存在且不是 placeholder；
- furniture collision-free；
- furniture in-bounds；
- Sionna / Mitsuba 可加载；
- room / furniture / material schema 合法；
- AP、RX、window、wall、furniture 等实体位置合法。

这一层回答的问题是：

> 这个场景能不能作为无线仿真的输入？

---

### Layer B：Network Simulation Validity

这一层判断网络仿真结果是否物理合理。

建议包含：

- RSS 不超过发射功率太多；
- RSS 在合理范围；
- BER 在 `[0, 1]`；
- throughput 不超过物理上限；
- Jain fairness 在 `[0, 1]`；
- 频率升高时 coverage 不应异常变好；
- 墙体、材料、遮挡应该对覆盖产生影响。

这一层回答的问题是：

> 输出的通信指标是不是物理上可信？

---

### Layer C：Task Completion

这一层判断具体 capability 是否真正完成。

例如：

- C1 有没有报告 `coverage_pct`；
- C2 有没有放置 2/3 个 AP；
- C3 有没有比较两个频率；
- C4 有没有 before / after；
- C5 有没有 CIR + QPSK + BER；
- C6 有没有 IRC aperture；
- C7 有没有 PF scheduling 和 per-user rate。

这一层回答的问题是：

> agent 是否真的完成了当前 capability 对应的下游任务？

---

## 3. 减少 `code_contains`，改成 JSON artifact 结构检查

当前很多 subcheck 是 grep 文本，例如：

- C3 检查 `<f1>_ghz`、`<f2>_ghz`；
- C7 检查 `pf_scheduling`；
- C6 检查 `irc_compliant`、`perimeter`、`aperture`。

这种方式容易被 agent 通过关键词糊弄过去。建议尽量改成 artifact schema check。

---

### C3 示例：频率比较

不建议只检查文本里有没有 `ghz`。  
建议要求 `simulation_result.json` 中显式包含：

```json
{
  "simulation_config": {
    "frequencies_ghz": [2.4, 5.0]
  },
  "numerical_metrics": {
    "coverage_pct_low_freq": 78.2,
    "coverage_pct_high_freq": 55.1,
    "coverage_diff_pp": 23.1
  }
}
```

verifier 检查：

- `frequencies_ghz` 长度为 2；
- 两个频率数值不同；
- `low_freq < high_freq`；
- 两个 coverage 字段都存在；
- `coverage_diff_pp ≈ coverage_low - coverage_high`；
- 高频 coverage 不应明显高于低频。

---

### C7 示例：PF scheduling

不建议只 grep `scheduling`。  
建议要求：

```json
{
  "system_config": {
    "num_cells": 3,
    "scheduler": "proportional_fair",
    "num_tti": 100,
    "num_users": 8
  },
  "numerical_metrics": {
    "mean_throughput_bps_hz": 4.8,
    "fairness_index": 0.82,
    "per_user_avg_rate": [3.1, 4.5, 5.0, 2.9, 4.2, 3.8, 5.1, 4.0]
  }
}
```

verifier 检查：

- `scheduler == "proportional_fair"`；
- `num_tti >= 100`；
- `len(per_user_avg_rate) == num_users`；
- fairness 由 `per_user_avg_rate` 重新计算后与报告值接近。

这样比 grep 强很多。

---

## 4. 增加数值一致性检查，防止只填合理数字

当前 oracle 已经检查物理范围，例如 RSS、BER、fairness、throughput。  
下一步建议加入 cross-field consistency，也就是字段之间必须互相对得上。

---

### C1：coverage 一致性

建议输出：

```json
{
  "coverage": {
    "num_grid_points": 400,
    "num_covered_points": 328,
    "threshold_dbm": -80
  },
  "numerical_metrics": {
    "coverage_pct": 82.0
  }
}
```

verifier 重新计算：

```text
coverage_pct = num_covered_points / num_grid_points * 100
```

如果 agent 报告的 `coverage_pct` 和 verifier 重算结果差太多，就失败。

---

### C2：multi-AP optimization 一致性

除了检查：

```text
min_rss_dbm >= -85
```

还建议检查：

- `len(ap_positions) == expected_num_aps`；
- AP 坐标在房间内；
- AP 高度合理，例如 2–3 m；
- AP 不能放在家具内部；
- AP 之间不能过近；
- `min_rss_dbm` 等于 reported RSS map 的最小值；
- multi-AP coverage 至少不比 single-AP baseline 差太多。

---

### C4：edit recompute 一致性

当前 C4 主要检查 before / after / delta 是否存在。  
建议改成输出多个 artifact：

- `scene_state_before.json`
- `scene_state_after.json`
- `simulation_result_before.json`
- `simulation_result_after.json`

verifier 检查：

```text
coverage_delta_pp = coverage_pct_after - coverage_pct_before
```

并且检查 scene edit 是否真的发生。

例如：

- 如果任务是 remove furniture，则 after scene 中该 furniture 应该不存在；
- 如果任务是 add partition，则 after scene 中应该新增 partition；
- 如果任务是 change material，则 material 字段应该发生变化。

---

### C7：system-level 一致性

Jain’s fairness 可以由 per-user rates 重新计算：

```text
J = (sum r_i)^2 / (n * sum r_i^2)
```

如果报告的 `fairness_index` 和 verifier 重算结果差太多，就失败。

同时可以检查：

- `mean_throughput_bps_hz` 是否等于 per-user 平均值；
- 每个用户是否至少被调度过一次；
- `per_user_avg_rate` 是否非负；
- `num_users` 和 rate 数组长度是否一致。

---

## 5. 给每个 capability 增加 baseline / relative improvement

当前很多 metric 是绝对阈值，例如：

- C2：`min_rss_dbm >= -85 dBm`
- C7：fairness ≥ 0.7 或 ≥ 0.6

绝对阈值简单，但不同房间、材料、频率、家具密度会导致难度差异很大。  
建议加入 relative improvement。

---

### C2：多 AP 优化

除了绝对阈值，还可以加入：

```text
optimized_min_rss_dbm >= baseline_min_rss_dbm + 3 dB
```

baseline 可以是：

- 随机 AP 放置；
- room centroid 放置；
- evenly-spaced AP placement。

这样可以判断 agent 是否真的做了优化，而不是随便放 AP。

---

### C4：场景编辑

可以根据 edit 类型定义 expected direction。

例如：

```json
{
  "edit": {
    "action": "add_partition",
    "expected_effect": "coverage_decrease_or_wall_spread_increase"
  }
}
```

一些合理规则：

- add partition：coverage 可能下降，或 per-room coverage spread 增大；
- remove large obstacle：coverage 通常上升；
- change wall material to concrete：coverage 通常下降；
- change wall material to wood / drywall：coverage 通常上升。

verifier 不必过于严格，但可以检查变化方向是否大致合理。

---

### C7：PF scheduling

可以比较 PF 与 baseline scheduler：

- PF fairness 应该高于 max-SINR；
- PF throughput 不一定最高，但不能低得离谱；
- PF 不应让用户 starvation；
- PF 的 per-user rate 分布应比 max-SINR 更均衡。

---

## 6. 把 C1–C7 调整成难度递进

现在 C1–C7 是 capability 分类。  
建议在论文或报告中进一步组织成三个阶段。

---

### Stage 1：RT Coverage Tasks

包括：

- C1 single AP coverage
- C2 multi-AP placement
- C3 material / frequency comparison
- C4 scene edit recomputation
- C6 IRC + coverage

这些任务主要验证：

> 3D scene 是否能支撑 ray-tracing-style coverage evaluation。

---

### Stage 2：PHY Coupling Tasks

包括：

- C5 RT-to-PHY

验证：

> agent 是否能从 scene / RT channel 进入通信物理层，例如 CIR、QPSK、BER。

---

### Stage 3：System-Level Tasks

包括：

- C7 multi-cell PF scheduling

验证：

> agent 是否能从单链路扩展到多用户、多小区、调度和公平性。

---

## 7. 明确区分真实仿真和轻量近似仿真

当前 verifier 允许 Sionna 不安装时跳过 `sionna_loadable_check`，并且 oracle 是根据 JSON 数值做纯 Python 检查。

这个设计适合大规模 benchmark，但论文里最好明确：

- agent 是否必须真的调用 Sionna RT；
- 是否允许 analytical approximation / path-loss model；
- verifier 是否只检查结果物理合理，而不强制仿真后端。

建议分成两个 setting。

---

### Approximate Mode

允许使用：

- path-loss model；
- shadowing；
- material attenuation；
- simplified wall-loss model；
- lightweight coverage grid simulation。

适合大规模自动评估。

---

### Executable RT Mode

要求：

- Sionna RT 能加载 scene；
- 使用真实或半真实 RT simulation；
- 输出 RSS map / CIR / path-level information；
- verifier 对 RT 输出做物理一致性检查。

适合小规模 high-fidelity subset。

可以在论文中表述为：

> The main benchmark uses lightweight executable verifiers for scalability, while a high-fidelity subset requires actual Sionna RT execution.

---

## 8. 各 capability 的具体修改建议

## C1：single AP coverage

当前任务是单 AP 覆盖率计算。建议保留，但加强以下检查：

- AP 必须接近 room centroid；
- AP 高度必须接近 2.5 m；
- AP 不能在墙体或家具内部；
- coverage 由 grid map 重算；
- 输出 `rss_grid_dbm` 或 RSS histogram；
- `coverage_pct` 必须和 covered grid points 一致。

建议 artifact 字段：

```json
{
  "network_entities": {
    "ap_position": [2.0, 2.0, 2.5]
  },
  "coverage": {
    "rx_height_m": 1.2,
    "grid_resolution_m": 0.25,
    "num_grid_points": 400,
    "num_covered_points": 328,
    "threshold_dbm": -80
  },
  "numerical_metrics": {
    "coverage_pct": 82.0
  }
}
```

---

## C2：multi-AP optimization

建议加强：

- 检查 AP 数量；
- 检查 AP 在房间内；
- 检查 AP 高度合理；
- 检查 AP 不能放在家具内部；
- 检查 AP 之间不能太近；
- 检查相对 single-AP 或 random baseline 有提升；
- 检查 per-room coverage 不完全相同；
- 检查 `min_rss_dbm` 是否来自 RSS map。

---

## C3：material_frequency

这是很有价值的任务，但目前 token check 偏弱。

建议改成：

- 强制输出 `frequencies_ghz: [low, high]`；
- 强制输出两个 coverage；
- verifier 重算 `coverage_diff_pp`；
- 检查 high frequency path loss 更大；
- 如果材料变化也在任务里，要求输出 material attenuation table；
- 检查 material trend 是否合理。

建议 artifact 字段：

```json
{
  "simulation_config": {
    "frequencies_ghz": [2.4, 5.0],
    "material_model": "wall_loss"
  },
  "numerical_metrics": {
    "coverage_pct_low_freq": 78.2,
    "coverage_pct_high_freq": 55.1,
    "coverage_diff_pp": 23.1
  }
}
```

---

## C4：scene_edit_recompute

这是非常重要的 capability，因为它测的是“场景变化后的因果更新”。

建议大幅加强：

- 输出 before / after 两个 scene；
- 输出 before / after 两个 result；
- verifier 检查 edit 是否真的发生；
- verifier 检查 delta 是否等于 after - before；
- 根据 edit 类型检查预期变化方向；
- 对 multi-room edit 检查 per-room coverage spread。

建议 artifact 文件：

```text
scene_state_before.json
scene_state_after.json
simulation_result_before.json
simulation_result_after.json
```

同时需要修正文档中的一个问题：  
C4 的 subcheck 数量前后不一致。正文写的是 10，但表格实际列到 11，summary 也写 11。建议统一为 11，或者检查是否有一个 subcheck 不该计入。

---

## C5：RT-to-PHY

建议从“BER 合法”升级为“PHY pipeline 合法”。

建议输出：

```json
{
  "phy_config": {
    "modulation": "QPSK",
    "snr_db": 10,
    "num_symbols": 10000
  },
  "channel": {
    "cir_path_count": 6,
    "path_delays_ns": [0.0, 12.4, 27.8],
    "path_gains_db": [-50.1, -63.2, -70.4]
  },
  "numerical_metrics": {
    "ber": 0.012,
    "ber_theoretical_awgn": 0.0008
  }
}
```

verifier 检查：

- path delays 非负；
- path gains 合理；
- `cir_path_count` 与 path arrays 长度一致；
- BER 在 `[0, 1]`；
- BER 随 SNR 增大下降；
- simulated BER 不应比 theoretical AWGN 好太多，因为 multipath 通常不会无代价地优于 AWGN baseline；
- 如果有 channel estimator，检查 NMSE 合理范围。

---

## C6：IRC + coverage

这个任务很有特色，因为它把建筑规范和无线覆盖结合起来。

建议强化：

- 明确 IRC R303 的 aperture ratio；
- 检查 `total_window_aperture_m2 >= 0.08 * floor_area_m2`；
- 检查 window 在 perimeter wall；
- 检查 window 不超出墙面边界；
- 检查 AP 不放在 window / wall / furniture 内；
- 检查 `irc_compliant` 是由数值规则推出来的，而不是随便填的 boolean。

建议 artifact 字段：

```json
{
  "building_code": {
    "code": "IRC_R303",
    "floor_area_m2": 20.0,
    "required_window_aperture_m2": 1.6,
    "total_window_aperture_m2": 1.8,
    "windows_on_perimeter": true,
    "irc_compliant": true
  },
  "numerical_metrics": {
    "coverage_pct": 74.5
  }
}
```

verifier 重新计算：

```text
required_window_aperture_m2 = 0.08 * floor_area_m2
```

并检查 total aperture 是否满足要求。

---

## C7：system-level multicell

这是最容易被“填数”糊弄的 capability。建议重点加强：

- 强制输出 `num_users`；
- 强制输出 `per_user_avg_rate`；
- verifier 重算 mean throughput；
- verifier 重算 Jain fairness；
- 检查 `num_tti >= 100`；
- 检查每个用户至少被调度过一次；
- 检查 PF metric，例如 `instantaneous_rate / average_rate`；
- 检查 per-cell load 是否合理；
- 检查 SINR 和 throughput 是否在物理范围内。

建议 artifact 字段：

```json
{
  "system_config": {
    "num_cells": 3,
    "num_users": 8,
    "num_tti": 100,
    "scheduler": "proportional_fair"
  },
  "scheduling_summary": {
    "per_user_scheduled_tti": [14, 11, 12, 10, 13, 15, 12, 13]
  },
  "numerical_metrics": {
    "mean_throughput_bps_hz": 4.2,
    "fairness_index": 0.86,
    "sinr_mean_db": 15.7,
    "per_user_avg_rate": [3.5, 4.1, 4.8, 3.9, 4.0, 4.4, 4.3, 4.6]
  }
}
```

verifier 检查：

```text
fairness_index = (sum r_i)^2 / (n * sum r_i^2)
mean_throughput_bps_hz = mean(per_user_avg_rate)
```

---

## 9. 推荐统一的 `simulation_result.json` schema

建议每个 task 都统一输出类似结构：

```json
{
  "task_id": "TC_C2_xxx",
  "capability": "multi_ap_optimization",
  "status": "completed",

  "scene_ref": "scene_state.json",

  "simulation_config": {
    "frequency_ghz": 5.0,
    "tx_power_dbm": 20,
    "grid_resolution_m": 0.25,
    "model_type": "path_loss_or_rt"
  },

  "network_entities": {
    "ap_positions": [
      {"id": "ap_1", "position": [1.5, 2.0, 2.5]},
      {"id": "ap_2", "position": [4.5, 2.0, 2.5]}
    ],
    "rx_grid": {
      "height_m": 1.2,
      "num_points": 400
    }
  },

  "raw_outputs": {
    "rss_grid_dbm": [],
    "per_room_coverage_pct": {}
  },

  "numerical_metrics": {
    "coverage_pct": 91.2,
    "min_rss_dbm": -78.4
  },

  "consistency": {
    "coverage_recomputed_from_grid": true,
    "artifact_schema_version": "v1"
  }
}
```

这个 schema 的好处是：

- verifier 更稳定；
- 方便后续分析失败案例；
- 可以统一 C1–C7 的输出结构；
- 可以支持 lightweight simulation 和 real Sionna RT 两种模式；
- 可以减少对 `code_contains` 的依赖。

---

## 10. 推荐最终 benchmark 叙述

可以把整个方案表述为：

> 我们提出一个面向无线网络的 3D 场景下游任务评测框架，不仅验证生成场景的几何可用性，还通过覆盖率、链路级 BER、建筑规范约束和多小区调度等任务，评估场景是否能作为可执行、物理可信的网络仿真基底。

英文版本：

> We evaluate whether generated 3D indoor scenes can serve as actionable substrates for downstream wireless-network tasks. Each task requires producing both a valid scene representation and a task-specific simulation result, with verification performed through geometry checks, physical plausibility checks, and capability-specific network metrics.

---

## 11. 最重要的修改清单

优先级从高到低：

1. **减少 grep-style check**  
   将 `code_contains` 尽量替换成 JSON schema check 和 numeric consistency check。

2. **增加 cross-field consistency**  
   例如 coverage 由 grid points 重算，fairness 由 per-user rate 重算，delta 由 after-before 重算。

3. **引入 baseline / relative improvement**  
   尤其适用于 C2、C4、C7。

4. **明确 approximate simulation vs real Sionna RT simulation**  
   主 benchmark 可以轻量，high-fidelity subset 要求真实 RT。

5. **修正文档中的 C4 数量不一致**  
   当前正文和 summary 对 C4 subcheck 总数不一致，应统一。

6. **强化 C3、C6、C7 的真实性检查**  
   C3 要检查真实频率数值；C6 要重算 aperture；C7 要重算 fairness 和 throughput。

---

## 12. 总结

这个方案的核心价值在于：

> 它不只是评估模型能否生成一个 3D 房间，而是评估生成的 3D 场景是否能支撑无线网络中的真实下游任务。

这比单纯的 scene generation benchmark 更有意义，也更容易形成独特贡献。

建议最终把贡献点组织为：

1. 3D scene generation as wireless simulation substrate；
2. C1–C7 downstream task suite；
3. Three-layer verifier: scene validity, network plausibility, task completion；
4. Lightweight oracle + optional high-fidelity Sionna RT mode；
5. JSON artifact consistency checks for scalable automatic evaluation。
