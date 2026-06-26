#!/usr/bin/env bash
# Poll the dual fix-check run; exit (notifying the agent) as soon as there is
# ENOUGH evidence (deadlock broken + val/nll below the 1.382 unigram floor),
# or the job ends, or timeout.
cd /lustre/scratch126/cellgen/lotfollahi/ha11/bd3lms
PY=/software/cellgen/team361/ha11/envs/nichejepa/bin/python
JID=${1:-50127}
for i in $(seq 1 40); do
  DIR=""
  for d in outputs/carbon-prokaryote/2026.06.18/*/; do
    [ -f "${d}.hydra/config.yaml" ] && grep -qE "^  length: 4608" "${d}.hydra/config.yaml" 2>/dev/null && DIR="$d"
  done
  OUT=""
  if [ -n "$DIR" ] && ls "${DIR}checkpoints/"*.ckpt >/dev/null 2>&1; then
    OUT=$($PY scripts/inspect_fix.py "${DIR%/}" 2>/dev/null | grep -vi warn)
    SCORE=$(echo "$OUT" | grep -oE "best_val_nll=[0-9.]+|best_val_nll=None" | head -1 | cut -d= -f2)
    BROKEN=$(echo "$OUT" | grep -c "BROKEN")
    echo "[poll $i $(date +%H:%M:%S)] dir=$DIR score=$SCORE broken=$BROKEN"
    if [ "$BROKEN" = "1" ] && [ -n "$SCORE" ] && [ "$SCORE" != "None" ]; then
      if awk "BEGIN{exit !($SCORE < 1.375)}"; then
        echo "=== ENOUGH_EVIDENCE: deadlock broken AND val/nll=$SCORE < 1.382 floor ==="
        echo "$OUT"; exit 0
      fi
    fi
  fi
  st=$(bjobs -noheader -o stat "$JID" 2>/dev/null)
  if [ -z "$st" ] || [ "$st" = "EXIT" ] || [ "$st" = "DONE" ]; then
    echo "=== JOB_ENDED stat=$st ==="; echo "$OUT"; tail -20 logs/fixcheck_dual_${JID}.out 2>/dev/null; exit 0
  fi
  sleep 45
done
echo "=== TIMEOUT ==="; echo "$OUT"