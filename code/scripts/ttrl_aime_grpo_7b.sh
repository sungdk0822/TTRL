set -x

echo "Starting Ray Cluster..."
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "Starting Model Training..."

ROOT_DIR=$1
MODEL_DIR=$2
WANDB_KTY=$3
# ------------------------------------------------------------

DATE=$(date +%m%d)
EXPERIMENT="TTRL"

ADVANTAGE="group_norm"
TASK="AIME-TTT"
BACKBONE="Qwen2.5-Math-7B"

mkdir -p ${ROOT_DIR}/logs
mkdir -p ${ROOT_DIR}/outputs

MODEL="${TASK}-${BACKBONE}"
BACKBONE_PATH="${MODEL_DIR}/${BACKBONE}"
OUTPUT_DIR="${ROOT_DIR}/outputs/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}"

# ------------------------------------------------------------

EXP="${DATE}-${TASK}-${EXPERIMENT}-${BACKBONE}-${ADVANTAGE}"
LOG_FILE="${ROOT_DIR}/logs/${EXP}.log"

ray job submit --address="http://localhost:8265" \
   --runtime-env-json='{"conda": "base"}' \
   -- python -m ttrl.cli.train_ppo_naive \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 1 \
   --colocate_actor_ref \
   --pretrain "${BACKBONE_PATH}" \
   --save_path "${OUTPUT_DIR}/model" \
   --verify_task "ttt" \
   --verify_task_eval "math" \
   --micro_train_batch_size 4 \
   --train_batch_size 16 \
   --num_episodes 60 \
   --save_steps 1 \
   --eval_steps 1 \
   --logging_steps 1 \
   --max_samples 400000 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 16 \
   --n_samples_per_prompt 16 \
   --n_votes_per_prompt 64 \
   --extra_eval_task "AMC-TTT,MATH-TTT" \
   --training_mode "rl" \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 3072 \
   --advantage_estimator ${ADVANTAGE} \
   --use_kl_loss \
   --temperature 1.0 \
   --eval_temperature 0.0 \
   --lambd 1.0 \
   --gamma 1.0 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.00 \
   --prompt_data "json@${ROOT_DIR}/data/${TASK}" \
   --input_key "prompt" \
   --label_key "answer" \
   --max_ckpt_num 30 \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --flash_attn \
   --use_wandb ${WANDB_KTY} \
   --wandb_project TTRL \
   --wandb_run_name ${EXP} \
   --use_tensorboard "${ROOT_DIR}/logs/${EXP}" \
   --ckpt_path "${OUTPUT_DIR}/ckpt"
#  > ${LOG_FILE} 2>&1 &

echo "Model Training started in background. Check logs at ${LOG_FILE}"
