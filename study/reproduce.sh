#!/usr/bin/env bash
# Regenerate study/REPRODUCTION.md: rerun the study's headline commands and record what they print.
#
# The point is to keep the "command -> measurement" chain of study/REPORT.md, study/README.md and
# study/NOTES.md verifiable at the current commit: every table in those files was produced by one of
# these commands, so if an output here stops matching the table, the drift is visible instead of
# silent.  Total runtime is roughly 15 minutes; each command is capped at 20 minutes.
set -u
cd "$(dirname "$0")/.."

out=study/REPRODUCTION.md
head=$(git rev-parse --short HEAD)

commands=(
  "python3 -m study.exp10_sr0_geometry"
  "python3 -m study.exp11_dh_continuation --steps 3"
  "python3 -m study.exp12_sr0_solver --poses 2 --seeds 300 --match 1e-3"
  "python3 -m study.exp02_pair_census --poses 2 --seeds 300 --resolve"
  "python3 -m study.exp15_branch_labels --poses 1 --seeds 400"
  "python3 -m study.exp19_partition --poses 1 --seeds 600 --waypoints 4"
  "python3 -m study.exp20_labels_redundant --robot franka --seeds 120"
  "python3 -m study.exp21_endgame --pose-seed 0"
  "python3 -m study.exp25_deterministic_count --pose-seed 0"
  "python3 -m study.exp30_adaptive_scan --pose-seeds 0 --uniform-grid 5 --uniform-seeds 80"
)

{
  echo "# 复现抽查（基线提交 $head）"
  echo
  echo "本文件记录在提交 \`$head\` 上重跑本课题十条头条命令时的实际输出（每个命令截取末尾若干行），"
  echo "用于确认 REPORT.md / README.md / NOTES.md 里“命令 → 测量”的链条在最终状态下仍然成立。"
  echo "重新生成：\`bash study/reproduce.sh\`（约 15 分钟）。"
  for command in "${commands[@]}"; do
    echo
    echo "## \`$command\`"
    echo
    echo '```'
    timeout 1200 $command 2>&1 | tail -n 8
    echo '```'
  done
} > "$out"

echo "wrote $out"
