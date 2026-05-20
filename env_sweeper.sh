#!/bin/sh
# workdir=$(cd $(dirname $0); pwd)
workdir=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
echo "workdir:" $workdir

export DATA_LOG=$workdir/logdir/b3_F32_FixMultiview
export DATA_CONFIG=$workdir/occdepth/config/sweeper/sp_flospdepth_crp_stereodepth_cascadecls.yaml
export PYTHONPATH=$workdir:$PYTHONPATH
export ETS_TOOLKIT=qt4
export QT_API=pyqt5
