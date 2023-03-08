#!/bin/bash

# set -vxeuo pipefail
set -e

error_msg="Specify '-it' or '-d' on the command line as a first argument."

arg="${1:-}"

if [ -z "${arg}" ]; then
    echo "${error_msg}"
    exit 1
elif [ "${arg}" != "-it" -a "${arg}" != "-d" ]; then
    echo "${error_msg} Specified argument: ${arg}"
    exit 2
fi

if [ "${arg}" == "-it" ]; then
    remove_container="--rm"
else
    remove_container=""
fi

SIREPO_SRDB_HOST="${SIREPO_SRDB_HOST:-}"
SIREPO_SRDB_GUEST="${SIREPO_SRDB_GUEST:-}"
SIREPO_SRDB_ROOT="${SIREPO_SRDB_ROOT:-'/sirepo'}"

unset cmd _cmd docker_image SIREPO_DOCKER_CONTAINER_ID

month=$(date +"%m")
day=$(date +"%d")
year=$(date +"%Y")

today="${HOME}/tmp/data/${year}/${month}/${day}"

if [ -d "${today}" ]; then
    echo "Directory ${today} exists."
else
    echo "Creating Directory ${today}"
    mkdir -p "${today}"
fi

# docker_image="radiasoft/sirepo:beta"
docker_image="radiasoft/sirepo:20220806.215448"
docker_binary=${DOCKER_BINARY:-"docker"}

${docker_binary} pull ${docker_image}

${docker_binary} images

in_docker_cmd="mkdir -v -p ${SIREPO_SRDB_ROOT} && \
    if [ ! -f "${SIREPO_SRDB_ROOT}/auth.db" ]; then \
        cp -Rv /SIREPO_SRDB_ROOT/* ${SIREPO_SRDB_ROOT}/; \
    else \
        echo 'The directory exists. Nothing to do'; \
    fi && \
    sirepo service http"
cmd_start="${docker_binary} run ${arg} --init ${remove_container} --name sirepo \
    -e SIREPO_AUTH_METHODS=bluesky:guest \
    -e SIREPO_AUTH_BLUESKY_SECRET=bluesky \
    -e SIREPO_SRDB_ROOT=${SIREPO_SRDB_ROOT} \
    -e SIREPO_COOKIE_IS_SECURE=false \
    -p 8000:8000 \
    -v $PWD/sirepo_bluesky/tests/SIREPO_SRDB_ROOT:/SIREPO_SRDB_ROOT:ro,z "

cmd_extra=""
if [ ! -z "${SIREPO_SRDB_HOST}" -a ! -z "${SIREPO_SRDB_GUEST}" ]; then
    cmd_extra="-v ${SIREPO_SRDB_HOST}:${SIREPO_SRDB_GUEST} "
fi

cmd_end="${docker_image} bash -l -c \"${in_docker_cmd}\""

cmd="${cmd_start}${cmd_extra}${cmd_end}"

echo -e "Command to run:\n\n${cmd}\n"
if [ "${arg}" == "-d" ]; then
    SIREPO_DOCKER_CONTAINER_ID=$(eval ${cmd})
    export SIREPO_DOCKER_CONTAINER_ID
    echo "Container ID: ${SIREPO_DOCKER_CONTAINER_ID}"
    ${docker_binary} ps -a
    ${docker_binary} logs ${SIREPO_DOCKER_CONTAINER_ID}
else
    eval ${cmd}
fi
