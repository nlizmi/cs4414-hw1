#!/bin/bash

DATA_DIR=./data
mkdir -p "${DATA_DIR}"

wget -O "${DATA_DIR}/passages_emb.json" https://cornell.box.com/shared/static/stwnjqkjs06rgs1512k0jqwst2bvx6ll.json
wget -O "${DATA_DIR}/queries_emb.json" https://cornell.box.com/shared/static/5xpdum6tuqx573kjrtcv3w1rbxdmqz4s.json
