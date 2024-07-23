
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib

CHECKPOINT_PATH=${CHECKPOINT_PATH:-/data/tattafos/gpt-j-checkpoint}
CALIBRATION_DATA_PATH=${CALIBRATION_DATA_PATH:-/data/tattafos/cnn_dailymail/calibration-data/cnn_dailymail_calibration.json}
NUM_GROUPS=128
NUM_SAMPLES=512
ITERS=200

python -u main.py \
	--model_name ${CHECKPOINT_PATH} \
	--dataset ${CALIBRATION_DATA_PATH} \
	--group_size ${NUM_GROUPS} \
	--bits 4 \
	--iters ${ITERS} \
	--nsamples ${NUM_SAMPLES} \
	--device cpu \
	--deployment_device "gpu" \
	--scale_dtype 'fp32' \
	--disable_eval \
	--output_dir moe-autoround-${NUM_GROUPS}g-${NUM_SAMPLES}nsamples-${ITERS}iters 2>&1 | tee autoround_log_${NUM_GROUPS}g_${NUM_SAMPLES}nsamples_${ITERS}iters.log



