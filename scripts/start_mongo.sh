#!/bin/bash

set -vxeuo pipefail

error_msg="Specify '-it' or '-d' on the command line as a first argument."

arg="${1:-}"

if [ -z "${arg}" ]; then
    echo "${error_msg}"
    exit 1
elif [ "${arg}" != "-it" -a "${arg}" != "-d" ]; then
    echo "${error_msg} Specified argument: ${arg}"
    exit 2
fi

docker_image="mongo"
docker_binary=${DOCKER_BINARY:-"docker"}

${docker_binary} pull ${docker_image}
${docker_binary} images
${docker_binary} run ${arg} --rm -p 27017:27017 --name mongo ${docker_image}
