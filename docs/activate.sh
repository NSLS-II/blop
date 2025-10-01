#!/usr/bin/env bash

set -eux

mkdir -p ${CONDA_PREFIX}/share/jupyter/lab/settings
cp ${PIXI_PROJECT_ROOT}/.binder/overrides.json ${CONDA_PREFIX}/share/jupyter/lab/settings/overrides.json
