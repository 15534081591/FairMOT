if [ $# != 2 ]; then
  echo "Usage: 
        bash run_standalone_train_gpu.sh [config_file] [pretrained_model]
       " 
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)

CONFIG=$(get_real_path $1)
echo "CONFIG: "$CONFIG

LOAD_PRE_MODEL=$(get_real_path $2)
echo "PRETRAINED_MODEL: "$LOAD_PRE_MODEL

if [ ! -f $CONFIG ]
then
    echo "error: config=$CONFIG is not a file."
exit 1
fi

if [ ! -f $LOAD_PRE_MODEL ]
then
    echo "error: pretrained_model=$LOAD_PRE_MODEL is not a file."
exit 1
fi

if [ -d "$BASE_PATH/../train" ];
then
    rm -rf $BASE_PATH/../train
fi
mkdir $BASE_PATH/../train
cd $BASE_PATH/../train || exit

export PYTHONPATH=${BASE_PATH}:$PYTHONPATH

echo "start training on single GPU"
env > env.log
echo
python -u ${BASE_PATH}/../train.py --device GPU --config_path $CONFIG \
  --load_pre_model ${LOAD_PRE_MODEL} &> train.log &
