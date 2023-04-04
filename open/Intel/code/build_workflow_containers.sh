#!/bin/env bash

export DOCKER_BUILD_ARGS="--build-arg ftp_proxy=${ftp_proxy} --build-arg FTP_PROXY=${FTP_PROXY} --build-arg http_proxy=${http_proxy} --build-arg HTTP_PROXY=${HTTP_PROXY} --build-arg https_proxy=${https_proxy} --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg no_proxy=${no_proxy} --build-arg NO_PROXY=${NO_PROXY} --build-arg socks_proxy=${socks_proxy} --build-arg SOCKS_PROXY=${SOCKS_PROXY}"

export DOCKER_RUN_ENVS="--env ftp_proxy=${ftp_proxy} --env FTP_PROXY=${FTP_PROXY} --env http_proxy=${http_proxy} --env HTTP_PROXY=${HTTP_PROXY} --env https_proxy=${https_proxy} --env HTTPS_PROXY=${HTTPS_PROXY} --env no_proxy=${no_proxy} --env NO_PROXY=${NO_PROXY} --env socks_proxy=${socks_proxy} --env SOCKS_PROXY=${SOCKS_PROXY}"

function usage() {
  echo "To build any of the workflow containers, try:"
  echo "$ bash build_workflow_containers.sh <WORKFLOW_NAME>"
  echo "Supported workflows are: bert-99.9_jpqd-large and retinanet"
  echo "To build all workflow containers try:"
  echo "$ bash build_workflow_containers.sh --all"
}

function build_workflow_container() {
  local WORKFLOW=$1
  local FRAMEWORK=$2
  pushd ${WORKFLOW}/${FRAMEWORK}-cpu/docker
  bash build_${WORKFLOW}_container.sh
  popd
}

WORKFLOW=$(echo $1 | tr '[:upper:]' '[:lower:]')
case ${WORKFLOW} in
  bert-99.9_jpqd-large)
    build_workflow_container ${WORKFLOW} openvino
    ;;
  retinanet)
    build_workflow_container ${WORKFLOW} pytorch
    ;;
  --all)
    WORKFLOWS=(bert-99.9_jpqd-large retinanet)
    FRAMEWORKS=(openvino pytorch)
    for i in "${!WORKFLOWS[@]}"; do
      build_workflow_container ${WORKFLOWS[i]} ${FRAMEWORKS[i]}
    done
    ;;
 *)
   usage;
   ;;
esac