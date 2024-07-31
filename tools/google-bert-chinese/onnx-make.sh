#!/bin/bash -l

script_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $script_path/../requirements.bash

set -e -o pipefail

GOOGLE_BERT_CHINESE_DIR=$script_path
GOOGLE_BERT_MODEL_DIR=$GOOGLE_BERT_CHINESE_DIR/base_model
GOOGLE_BERT_MODEL_ONNX_FILE=$GOOGLE_BERT_MODEL_DIR/bert_base_chinese.onnx
GOOGLE_BERT_ONNX_DIR=$GOOGLE_BERT_CHINESE_DIR/onnx-artifacts

GOOGLE_BERT_PYVENV_NAME=google-bert-chinese
GOOGLE_BERT_EXPORT_PYVENV_NAME=google-bert-chinese-export

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

# 废弃
function onnx-training {
    workon_pyenv_or_create $GOOGLE_BERT_PYVENV_NAME

    pip install --require-virtualenv -r $GOOGLE_BERT_CHINESE_DIR/requirements.txt
    info "OK. installed requirements.txt for $GOOGLE_BERT_PYVENV_NAME."

    rm -rf $GOOGLE_BERT_ONNX_DIR && mkdir -p $GOOGLE_BERT_ONNX_DIR
    python $GOOGLE_BERT_CHINESE_DIR/../onnxruntime-training-artifacts.py $GOOGLE_BERT_MODEL_DIR/*.onnx $GOOGLE_BERT_ONNX_DIR
    info "Done. model onnx ok."

    deactivate
}

function official-model-export {
    workon_pyenv_or_create $GOOGLE_BERT_EXPORT_PYVENV_NAME

    pip install optimum[exporters] accelerate

    rm -rf $GOOGLE_BERT_MODEL_DIR && mkdir -p $GOOGLE_BERT_MODEL_DIR
    optimum-cli export onnx \
        --model google-bert/bert-base-chinese \
        --opset 14 \
        --batch_size 4 \
        --sequence_length 256 \
        $GOOGLE_BERT_MODEL_DIR
    info "Done. optimum-cli export onnx ok."

    mv $GOOGLE_BERT_MODEL_DIR/*.onnx $GOOGLE_BERT_MODEL_ONNX_FILE || return
    export opt_onnx_model=$GOOGLE_BERT_MODEL_ONNX_FILE

    # deactivate
}

function git_model-bert-base-chinese {
    info "git clone https://hf-mirror.com/google-bert/bert-base-chinese"
    # GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/google-bert/bert-base-chinese $GOOGLE_BERT_MODEL_DIR
    if [ -d $GOOGLE_BERT_MODEL_DIR ]; then
        git clean -xfd && git pull
        return
    fi

    rm -rf $GOOGLE_BERT_MODEL_DIR
    GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/google-bert/bert-base-chinese $GOOGLE_BERT_MODEL_DIR
    (cd $GOOGLE_BERT_MODEL_DIR && git lfs pull)
    info "Ok."
}

function workon_venv {
    workon_pyenv_or_create $GOOGLE_BERT_PYVENV_NAME

    md5_venv=$(cat ${VIRTUAL_ENV}/requirements.txt.md5 || echo "")
    md5_req=$(md5sum $GOOGLE_BERT_CHINESE_DIR/requirements.txt|awk '{print $1}')
    if [[ "x$md5_venv" == "x$md5_req"  ]]; then
        return
    fi

    pip install --require-virtualenv -r $GOOGLE_BERT_CHINESE_DIR/requirements.txt
    info "OK. installed requirements.txt for $GOOGLE_BERT_PYVENV_NAME."
    # echo $md5_req > ${VIRTUAL_ENV}/requirements.txt.md5

    # MAGIC_ONNX=GOOGLE_BERT_MODEL_DIR=$GOOGLE_BERT_CHINESE_DIR/MagicONNX
    # git clone https://gitee.com/Ronnie_zheng/MagicONNX.git $MAGIC_ONNX
    # (cd $MAGIC_ONNX && git checkout dev && pip3 install .)

    echo $md5_req > ${VIRTUAL_ENV}/requirements.txt.md5
}
# 输入参数：${model_dir} ${output_path} ${seq_length} 
# python3 pth2onnx.py ./bert-base-chinese ./bert_base_chinese.onnx 384
function onnx-export {
    workon_venv

    model_dir=$GOOGLE_BERT_MODEL_DIR
    output_path=$GOOGLE_BERT_MODEL_ONNX_FILE
    seq_length=256

    python $GOOGLE_BERT_CHINESE_DIR/onnx-export.py $model_dir $output_path $seq_length
    info "Done. onnx-export ok."

    export opt_onnx_model=$output_path

    # deactivate
}

# 修改优化模型：${bs}:[1, 4, 8, 16, 32, 64],${seq_len}:384
function onnx-optimum {
    workon_venv

    bs=16
    seq_len=256

    opt_onnx_model=${GOOGLE_BERT_MODEL_ONNX_FILE}.${bs}.onnx
    if [ -f $opt_onnx_model ]; then
        rm -rf $opt_onnx_model
    fi
    python3 -m onnxsim $GOOGLE_BERT_MODEL_ONNX_FILE $opt_onnx_model \
        --no-large-tensor \
        --overwrite-input-shape "input_ids:${bs},${seq_len}" "attention_mask:${bs},${seq_len}" "token_type_ids:${bs},${seq_len}"

    export opt_onnx_model
    info "Done. onnx-optimum ok."
}

function onnx-artifacts {
    workon_venv

    info "Use mode: ${1:-$opt_onnx_model}"
    
    rm -rf $GOOGLE_BERT_ONNX_DIR && mkdir -p $GOOGLE_BERT_ONNX_DIR
    python3 $GOOGLE_BERT_CHINESE_DIR/onnx-artifacts.py ${1:-$opt_onnx_model} $GOOGLE_BERT_ONNX_DIR
    info "Done. onnx-artifacts ok."
}

# if [ $# -lt 1 ]; then
#     model-export
#     onnx-training
#     exit 0
# fi

if [ $# -lt 1 ]; then
    # official-model-export
    workon_venv
    git_model-bert-base-chinese
    onnx-export
    onnx-optimum
    onnx-artifacts
    deactivate
fi

SUBCMD=$1
shift

command -v $SUBCMD > /dev/null || die "subcommand $SUBCMD required, not exists function $SUBCMD at the helper.sh ."

$SUBCMD $@

