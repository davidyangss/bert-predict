#!/bin/bash -l

script_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $script_path/../requirements.bash

set -e -o pipefail

GOOGLE_BERT_CHINESE_DIR=$script_path
GOOGLE_BERT_MODEL_DIR=$GOOGLE_BERT_CHINESE_DIR/base_model
GOOGLE_BERT_MODEL_ONNX_FILE=$GOOGLE_BERT_MODEL_DIR/model.onnx
GOOGLE_BERT_ONNX_DIR=$GOOGLE_BERT_CHINESE_DIR/onnx-artifacts

GOOGLE_BERT_PYVENV_NAME=google-bert-chinese-onnx
GOOGLE_BERT_EXPORT_PYVENV_NAME=hfoptimum

SHAPE_BATCH_SIZE=4
SHAPE_SEQ_LEN=256
OPSET_VERSION=14

function workon_pyenv_or_create {
    pyenv_name=$1
    if [ "$VIRTUAL_ENV_PROMPT" = "$pyenv_name" ]; then
        info "OK. workon $pyenv_name."
        return
        
    fi
    if workon $pyenv_name > /dev/null; then
        info "OK. workon $pyenv_name."
        return
    fi

    command -v python3 > /dev/null || die "python3 required, but it is not installed."
    command -v virtualenv > /dev/null || die "virtualenv required, but it is not installed."

    mkvirtualenv $pyenv_name || die "mkvirtualenv $pyenv_name failed."
    info "OK. created env $pyenv_name."

    workon $pyenv_name
}

# function venv_hfoptimum {
#     workon_pyenv_or_create $GOOGLE_BERT_EXPORT_PYVENV_NAME
#     pip install optimum[exporters] accelerate
#     python -m pip install optimum
#     pip install --upgrade --upgrade-strategy eager optimum[onnxruntime]
# }

function venv_hfoptimum {
    workon_pyenv_or_create $GOOGLE_BERT_EXPORT_PYVENV_NAME

    requirements_txt=$GOOGLE_BERT_CHINESE_DIR/hfoptimum-requirements.txt

    md5_venv=$(cat ${VIRTUAL_ENV}/requirements.txt.md5 || echo "")
    md5_req=$(md5sum $requirements_txt|awk '{print $1}')
    if [[ "x$md5_venv" == "x$md5_req"  ]]; then
        return
    fi

    pip install --require-virtualenv -r $requirements_txt
    info "OK. installed requirements.txt for $GOOGLE_BERT_EXPORT_PYVENV_NAME."
    echo $md5_req > ${VIRTUAL_ENV}/requirements.txt.md5
}

function official-model-export {
    venv_hfoptimum

    rm -rf $GOOGLE_BERT_MODEL_DIR && mkdir -p $GOOGLE_BERT_MODEL_DIR
    optimum-cli export onnx \
        --model google-bert/bert-base-chinese \
        --opset ${OPSET_VERSION} \
        --batch_size ${SHAPE_BATCH_SIZE} \
        --sequence_length ${SHAPE_SEQ_LEN} \
    info "Done. optimum-cli export onnx ok."

    mv $GOOGLE_BERT_MODEL_DIR/*.onnx $GOOGLE_BERT_MODEL_ONNX_FILE || return
    export opt_onnx_model=$GOOGLE_BERT_MODEL_ONNX_FILE

    deactivate
}

function git_model-bert-base-chinese {
    info "git clone https://hf-mirror.com/google-bert/bert-base-chinese"
    # GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/google-bert/bert-base-chinese $GOOGLE_BERT_MODEL_DIR
    if [ -d $GOOGLE_BERT_MODEL_DIR ]; then
        (cd $GOOGLE_BERT_MODEL_DIR && git clean -xfd && git pull)
        return
    fi

    rm -rf $GOOGLE_BERT_MODEL_DIR
    GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/google-bert/bert-base-chinese $GOOGLE_BERT_MODEL_DIR
    (cd $GOOGLE_BERT_MODEL_DIR && git lfs pull)
    info "Ok."
}

function venv_onnx {
    workon_pyenv_or_create $GOOGLE_BERT_PYVENV_NAME

    requirements_txt=$GOOGLE_BERT_CHINESE_DIR/onnx-requirements.txt

    md5_venv=$(cat ${VIRTUAL_ENV}/requirements.txt.md5 || echo "")
    md5_req=$(md5sum $requirements_txt|awk '{print $1}')
    if [[ "x$md5_venv" == "x$md5_req"  ]]; then
        return
    fi

    pip install --require-virtualenv -r $requirements_txt
    info "OK. installed requirements.txt for $GOOGLE_BERT_PYVENV_NAME."
    # echo $md5_req > ${VIRTUAL_ENV}/requirements.txt.md5

    # MAGIC_ONNX=GOOGLE_BERT_MODEL_DIR=$GOOGLE_BERT_CHINESE_DIR/MagicONNX
    # git clone https://gitee.com/Ronnie_zheng/MagicONNX.git $MAGIC_ONNX
    # (cd $MAGIC_ONNX && git checkout dev && pip3 install .)

    echo $md5_req > ${VIRTUAL_ENV}/requirements.txt.md5
}
# 输入参数：${model_dir} ${output_path} ${seq_length} ${batch_size} 
# python3 pth2onnx.py ./bert-base-chinese ./bert_base_chinese.onnx 384
function onnx-export-base_model {
    venv_hfoptimum

    model_dir=$GOOGLE_BERT_MODEL_DIR
    output_path=$GOOGLE_BERT_MODEL_ONNX_FILE

    batch_size=${SHAPE_BATCH_SIZE}
    seq_length=${SHAPE_SEQ_LEN}

    python $GOOGLE_BERT_CHINESE_DIR/model.onnx-export.py $model_dir $output_path $seq_length $batch_size
    info "Done. onnx-export ok."

    export opt_onnx_model=$output_path

    # deactivate
}

# 修改优化模型：${bs}:[1, 4, 8, 16, 32, 64],${seq_len}:384
function onnx-onnxsim {
    venv_hfoptimum

    bs=${SHAPE_BATCH_SIZE}
    seq_len=${SHAPE_SEQ_LEN}

    opt_onnx_model=${GOOGLE_BERT_MODEL_ONNX_FILE}.${bs}
    if [ -f $opt_onnx_model ]; then
        rm -rf $opt_onnx_model
    fi
    python3 -m onnxsim $GOOGLE_BERT_MODEL_ONNX_FILE $opt_onnx_model \
        --no-large-tensor \
        --overwrite-input-shape "input_ids:${bs},${seq_len}" "attention_mask:${bs},${seq_len}" "token_type_ids:${bs},${seq_len}"

    export opt_onnx_model
    info "Done. onnx-optimum ok."
}


function hfoptimum-export-model-onnx {
    venv_hfoptimum

    rm -rf $GOOGLE_BERT_MODEL_DIR && mkdir -p $GOOGLE_BERT_MODEL_DIR
    ONNX_OUTPUT=$GOOGLE_BERT_MODEL_DIR \
        ONNX_OPSET=$OPSET_VERSION \
        ONNX_BATCH_SIZE=$SHAPE_BATCH_SIZE \
        ONNX_SEQUENCE_LENGTH=$SHAPE_SEQ_LEN \
        python3 $GOOGLE_BERT_CHINESE_DIR/hfoptimum-model.py
    info "Done. hfoptimum-export-model-onnx ok."
}

function hfoptimum-check {
    venv_hfoptimum

    MODEL_PATH=$GOOGLE_BERT_MODEL_DIR \
        python3 $GOOGLE_BERT_CHINESE_DIR/hfoptimum-check.py
    info "Done. hfoptimum-check ok."
}

function hfoptimum-glue {
    venv_hfoptimum

    rm -rf $GOOGLE_BERT_MODEL_DIR && mkdir -p $GOOGLE_BERT_MODEL_DIR
    python3 $GOOGLE_BERT_CHINESE_DIR/hfoptimum-glue.py \
        --model_name_or_path google-bert/bert-base-chinese \
        --task_name sst2 \
        --optimization_level 1 \
        --overwrite_output_dir \
        --output_dir $GOOGLE_BERT_MODEL_DIR
    info "Done. hfoptimum-glue ok."
}


function onnx-artifacts {
    venv_onnx

    rm -rf $GOOGLE_BERT_ONNX_DIR && mkdir -p $GOOGLE_BERT_ONNX_DIR
    ONNX_MODEL=$GOOGLE_BERT_MODEL_DIR/model.onnx \
        ONNX_OUTPUT=$GOOGLE_BERT_ONNX_DIR \
        python3 $GOOGLE_BERT_CHINESE_DIR/onnx-artifacts.py
    info "Done($?). onnx-artifacts ok. $GOOGLE_BERT_MODEL_DIR/model.onnx ==> $GOOGLE_BERT_ONNX_DIR"
}

# if [ $# -lt 1 ]; then
#     model-export
#     onnx-training
#     exit 0
# fi

if [ $# -lt 1 ]; then
    (
        venv_hfoptimum
        hfoptimum-export-model-onnx
        hfoptimum-check
        info "Done. exported"
        deactivate
    )
    
    (
        venv_onnx
        info "onnx-artifacts..."
        onnx-artifacts
        info "Done($?). onnx-artifacts"
        deactivate
    )
    exit 0
fi

SUBCMD=$1
shift

command -v $SUBCMD > /dev/null || die "subcommand $SUBCMD required, not exists function $SUBCMD at the helper.sh ."

$SUBCMD $@

