#!/bin/sh
ARCH=$1

if [ $# -ne 1 ]
  then
    echo "Arguments error: <ARCH>"
    exit 1
fi

python -u examples/cluster.py -d ams --scale '30k' -a ${ARCH} -b 64 --width 640 --height 480
