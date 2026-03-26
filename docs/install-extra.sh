###########################################################################################
#                                                                                         #
# Note: if this script is updated make sure to update the copy in xsoft's home directory! #
#                                                                                         #
###########################################################################################

set -euxo pipefail

source /usr/share/Modules/init/bash

# Always install into the current environment
module load exfel pixi
module load exfel exfel-python

pixi update euxfel-extra --manifest-path ${PIXI_PROJECT_ROOT}
pixi install --manifest-path ${PIXI_PROJECT_ROOT}
