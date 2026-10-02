#!/usr/bin/env bash
# Regenerate study/REPRODUCTION.md: rerun the headline commands and record what they actually print.
#
# The point is to keep the "command -> measurement" chain of study/README.md and study/NOTES.md
# verifiable at the current commit: every table in those files was produced by one of these commands,
# so if an output here stops matching the table, the drift is visible instead of silent.
set -u
cd "$(dirname "$0")/.."

out=study/REPRODUCTION.md
head=$(git rev-parse --short HEAD)

{
  echo "# 复现抽查（基线提交 $head）"
  echo
  echo "本文件记录在提交 \`$head\` 上重跑关键命令时的实际输出（每个命令截取末尾 6 行），"
  echo "用于确认 README/NOTES 里“命令 → 测量”的链条在最终状态下仍然成立。"
  echo "重新生成：\`bash study/reproduce.sh\`（约 5~8 分钟）。"
  for command in \
    "python3 -m study.exp10_sr0_geometry" \
    "python3 -m study.exp12_sr0_solver --poses 2 --seeds 300 --match 1e-3" \
    "python3 -m study.exp21_endgame --pose-seed 0" \
    "python3 -m study.exp25_deterministic_count --pose-seed 0"; do
    echo
    echo "## \`$command\`"
    echo
    echo '```'
    timeout 900 $command 2>&1 | tail -n 6
    echo '```'
  done
} > "$out"

echo "wrote $out"
