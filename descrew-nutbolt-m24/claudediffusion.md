# claudediffusion

> 诊断日期: 2026-04-14 | 分支: diffusion | 模型: claude-sonnet-4-6

---

## 1. Current official plan

**PLANS_v3** 已于 2026-04-14 激活。目标：找到通过统一鲁棒性门控的 diffusion 候选，
或明确宣布暂停 diffusion 扩展。

- **Mainline A**: Robust-first latent tuning（最多 3 个候选）
- **Mainline B**: Residual-corrective re-entry（A 全部失败后触发）
- **硬止损**: 同方向连续 3 次失败 → 冻结该方向
- **门控**: single-seed `delta_hard >= -0.05`; multi-seed `hard_mean_delta >= 0`
- **冻结参考**: `run_a_latent_tailcoef02_thr015_sel_anchor003_seed42_15min`
  - nominal 1.675 / light_v2 1.638 / hard 1.505（seed 42）

---

## 2. Current diffusion status

| 路径 | 文件 | 状态 |
|------|------|------|
| Latent diffusion | `diffusion_latent_student.py` | **Active mainline** |
| Action-chunk diffusion | `diffusion_action_chunk_student.py` | 降级为 appendix |
| Residual diffusion | latent student mode 参数控制 | V3 备用，未触发 |

当前激活路径: `DiffusionLatentStudent`，backbone 冻结，仅训练 diffusion head。

---

## 3. Main failure source

**结论: 混合问题（mixed）**，权重不均等：

```
代码实现问题 (HIGH) > 训练/评估 gap (MEDIUM) > 方向/算法问题 (LOW)
```

方向本身（latent diffusion）理论可行，不是根本错误。
存在可验证的实现 bug，直接污染 loss 信号。
训练时指标虚高（1786 vs eval 1.49）说明 train/eval 分布 gap，但属次要问题。

---

## 4. High-confidence findings

**F1: 训练指标虚高，eval 指标真实且差**
- 训练时 diffusion latent best: 1786（高于 PAdapt 1496）
- 固定步数 eval: diffusion latent 1.49 vs PAdapt 1.92
- 多 seed 鲁棒性（seeds 42/43/44）:
  - PAdapt hard: 1.838 ± 0.105
  - Diffusion latent hard: 1.572 ± 0.192（delta = -0.266，显著差）
- 结论有数据支撑，不是主观判断

**F2: teacher_mu 未在生成时 clamp** [LOW → 已修复]
- 文件: `dexscrew/algo/ppo/diffusion_latent_student.py:666`
- `teacher_mu` 原始输出未 clamp，但 line 704 和 line 720 都使用 clamped 版本
- 实际影响较小（均在 no_grad 块内），但为一致性已修复

**F3: pred_latent 存在 double-tanh** [HIGH → 已修复]
- 文件: `diffusion_latent_student.py:698` + `models.py:129`
- `e_gt` 已在 `models.py:129` 中过 tanh，`pred_latent = torch.tanh(x0_pred)` 再过一次
- inference 路径 line 277 同样 double-tanh
- tanh(tanh(x)) 严重压缩分布尾部，降低 latent 表达力
- 已修复：移除 line 698 和 line 277 的 tanh

**F4: 训练稳定，无 NaN/梯度爆炸**
- 所有 run 正常完成，FPS 稳定（700-900）
- 问题不是训练崩溃，而是 loss 信号质量问题

---

## 5. Likely but unproven hypotheses

**H1**: 冻结 backbone 限制 latent 空间质量
- Diffusion head 在冻结 latent 空间上学习去噪
- 若 PAdapt latent 空间分布不规则/多峰，diffusion 无法改善
- 未验证: 缺少 latent 空间可视化（PCA/t-SNE）

**H2**: `teacher_delta_tail` loss 权重/目标设计可能有害
- 该 loss 是 V2 引入的辅助项，若目标本身有噪声（因 F2），
  可能主导了错误优化方向
- 未验证: 需要消融实验（coef=0 对比）

**H3**: 10 步 diffusion 推理步数可能不足
- 对复杂 latent 分布，10 步 DDPM 采样质量可能不足
- 未验证: 缺少不同推理步数的 eval 对比

---

## 6. Priority fixes

| 优先级 | 修复内容 | 层级 | 风险 |
|--------|----------|------|------|
| P1 | `diffusion_latent_student.py:666` 对 `teacher_mu` 加 clamp(-1,1) | execution | LOW |
| P2 | `:698` 移除 `pred_latent` 的 tanh（或对 target 也加 tanh 保持一致） | execution | MEDIUM |
| P3 | 修复 P1/P2 后，同 seed 重跑一个 V3-M1 候选对比 delta | execution | LOW |
| P4 | 对 latent 空间做 PCA/t-SNE 可视化，判断是否适合 diffusion | plan-level | LOW |
| P5 | 消融 `teacher_delta_tail` loss（coef=0），验证是否有害 | execution | MEDIUM |

---

## 7. Recommendation for next step

**建议: 先修 P1/P2 实现 bug，再继续 V3 候选实验**

理由:
- P1（teacher_mu clamp）是高置信度 bug，修复成本极低（1行），
  但可能直接影响 loss 信号质量
- P2（double tanh）修复后 latent 分布更宽，可能改善采样质量
- 在 bug 未修复的情况下继续跑 V3 候选，等于在有噪声的 loss 上做超参搜索
- 修复后若仍不过门控 → 再考虑是否修订 V3 方向（H1/H2 假设）

**不建议**:
- 现在切换到 action-chunk diffusion 主线（证据不足）
- 现在宣布 diffusion 失败（bug 未修复前结论不可靠）
- 大规模重构（超出当前诊断范围）
