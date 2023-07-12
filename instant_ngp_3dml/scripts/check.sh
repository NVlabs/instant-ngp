#!/bin/bash

ROOT_DIR=$(dirname ${0})
cd $ROOT_DIR/../..

3dml_check all . --config_json .vscode/config.json

result=$?
if [ $result -eq 0 ]
then
    echo "All is OK"
else
    echo "Please check errors"
fi

exit $result
