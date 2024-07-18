#!/bin/bash -l

script_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $script_path/../requirements.bash

set -e -o pipefail

GOOGLE_BERT_CHINESE_DIR=$script_path
GOOGLE_BERT_MODEL_DIR=$GOOGLE_BERT_CHINESE_DIR/model
GOOGLE_BERT_ONNX_DIR=$GOOGLE_BERT_CHINESE_DIR/onnx-training

GOOGLE_BERT_PYVENV_NAME=google-bert-chinese-training
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

function onnx-training {
    workon_pyenv_or_create $GOOGLE_BERT_PYVENV_NAME

    pip install --require-virtualenv -r $GOOGLE_BERT_CHINESE_DIR/requirements.txt
    info "OK. installed requirements.txt for $GOOGLE_BERT_PYVENV_NAME."

    rm -rf $GOOGLE_BERT_ONNX_DIR && mkdir -p $GOOGLE_BERT_ONNX_DIR
    python $GOOGLE_BERT_CHINESE_DIR/../onnxruntime-training-artifacts.py $GOOGLE_BERT_MODEL_DIR/*.onnx $GOOGLE_BERT_ONNX_DIR
    info "Done. model onnx ok."

    deactivate
}

function model-export {
    workon_pyenv_or_create $GOOGLE_BERT_EXPORT_PYVENV_NAME

    pip install optimum[exporters]

    rm -rf $GOOGLE_BERT_MODEL_DIR && mkdir -p $GOOGLE_BERT_MODEL_DIR
    optimum-cli export onnx --task text-classification --model google-bert/bert-base-chinese $GOOGLE_BERT_MODEL_DIR
    info "Done. optimum-cli export onnx ok."

    deactivate
}

if [ $# -lt 1 ]; then
    model-export
    onnx-training
    exit 0
fi

SUBCMD=$1
shift

command -v $SUBCMD > /dev/null || die "subcommand $SUBCMD required, not exists function $SUBCMD at the helper.sh ."

$SUBCMD $@

