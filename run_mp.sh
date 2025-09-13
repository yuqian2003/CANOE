set -euo pipefail

cd "$(dirname "$0")"

reset_mp() {
  pushd data >/dev/null
  unzip -o MP.zip >/dev/null
  mkdir -p MP
  mv -f train.csv MP/
  mv -f test.csv MP/
  popd >/dev/null
}

mkdir -p saved_models
cd saved_models
mkdir MP
cd ..

LR=0.05
ENTROPY_LOW=0.40
CHAOS_LIST=0.75,0.8,0.85,0.9

echo "=== CNOLP Training Start ==="

reset_mp
python model/run.py \
  --dataset MP \
  --topic 150 \
  --sequence_length 20 \
  --dim 8 \
  --at osc \
  --encoder trans \
  --type cnoa_mp \
  --bandwidth 1.0 \
  --batch 256 \
  --epoch 100 \
  --test_epoch 100 \
  --lr ${LR} \
  --entropy_low ${ENTROPY_LOW} \
  --chaotic_high_list ${CHAOS_LIST}

echo "CNOLP Training Completed."
