set -euo pipefail

cd "$(dirname "$0")"

reset_tc() {
  pushd data >/dev/null
  unzip -o TC.zip >/dev/null
  mkdir -p TC
  mv -f train.csv TC/
  mv -f test.csv TC/
  popd >/dev/null
}


mkdir -p saved_models
cd saved_models
mkdir TC
cd ..

LR=0.005
ENTROPY_LOW=0.40
CHAOS_LIST=0.75,0.8,0.85,0.9

echo "=== CANOE Training Start ==="

reset_tc
python model/run.py \
  --dataset TC \
  --topic 350 \
  --sequence_length 20 \
  --dim 16 \
  --at osc \
  --encoder trans \
  --type cnoa_tc \
  --bandwidth 1.0 \
  --batch 256 \
  --epoch 100 \
  --test_epoch 100 \
  --lr ${LR} \
  --entropy_low ${ENTROPY_LOW} \
  --chaotic_high_list ${CHAOS_LIST}

echo "CANOE Training Completed."



