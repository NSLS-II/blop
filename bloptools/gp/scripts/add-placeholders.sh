#!/bin/bash

set -e -o pipefail

for d in $(ls -1); do
    echo ${d}
    if [ -d "${d}" ]; then
        echo "${d} is a directory"

        content=$(ls -A ${d})
        echo "content of ${d} before: ***$content***"
        if [ -z "${content}" ]; then
            touch ${d}/.placeholder
        fi
        content=$(ls -A ${d})
        echo "content of ${d} after: ***$content***"

        if [ -d "${d}/lib" ]; then
            echo "${d}/lib exists"
            # ls -al ${d}/lib/
            content=$(ls -A ${d}/lib/)
            echo "content of ${d}/lib before: ***$content***"
            if [ -z "${content}" ]; then
                touch ${d}/lib/.placeholder
            fi
            content=$(ls -A ${d}/lib/)
            echo "content of ${d}/lib after: ***$content***"
        fi
    fi
done
