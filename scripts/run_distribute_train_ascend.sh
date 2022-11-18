echo "========================================================================"
echo "Please run the script as: "
echo "bash run_distribute_train_ascend.sh RANK_SIZE DATA_CFG(options) LOAD_PRE_MODEL(options)"
echo "For example: bash run_distribute_train_ascend.sh 8 ./data.json ./dla34.ckpt"
echo "It is better to use the absolute path."
echo "========================================================================"
set -e

RANK_SIZE=$1
DATA_CFG=$2
LOAD_PRE_MODEL=$3
export RANK_SIZE

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"

test_dist_8pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
    export RANK_SIZE=8
}

test_dist_2pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_2pcs.json
    export RANK_SIZE=2
}

test_dist_${RANK_SIZE}pcs

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ../
rm -rf distribute_train
mkdir distribute_train
cd distribute_train
for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cd ./device$i
    mkdir src
    cd src
    mkdir utils
    cd ../../../
    cp ./default_config.yaml ./distribute_train/device$i
    cp ./train.py ./distribute_train/device$i
    cp ./src/*.py ./distribute_train/device$i/src
    cp ./src/utils/*.py ./distribute_train/device$i/src/utils
    cd ./distribute_train/device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    if [ -f ${DATA_CFG} ]
    then
        python train.py --device Ascend --run_distribute True --data_cfg ${DATA_CFG} --load_pre_model ${LOAD_PRE_MODEL} --is_modelarts False > train$i.log 2>&1 &
    else
        python train.py --device Ascend --run_distribute True --is_modelarts False > train$i.log 2>&1 &
    fi
    echo "$i finish"
    cd ../
done

if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
echo "finish"
cd ../
