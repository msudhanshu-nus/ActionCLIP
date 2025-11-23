#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

type=$(python -c "import yaml;print(yaml.load(open('${config}'))['network']['type'])")
arch=$(python -c "import yaml;print(yaml.load(open('${config}'))['network']['arch'])")
dataset=$(python -c "import yaml;print(yaml.load(open('${config}'))['data']['dataset'])")
now=$(date +"%Y%m%d_%H%M%S")
output_root=${OUTPUT_ROOT:-exp}
log_root=${LOG_ROOT:-}

mkdir -p "${output_root}/${type}/${arch}/${dataset}/${now}"
if [ -n "${log_root}" ]; then
  mkdir -p "${log_root}"
  log_root_flag=(--log_root "${log_root}")
else
  log_root_flag=()
fi

python -u test.py --config ${config} --log_time $now --output_root "${output_root}" "${log_root_flag[@]}" \
  2>&1 | tee "${output_root}/${type}/${arch}/${dataset}/${now}/$now.log"
# --mail-user=mengmengwang@zju.edu.cn --mail-type=ALL
