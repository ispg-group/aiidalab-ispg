#!/bin/bash

# This script automatically creates required code Nodes in AiiDA database,
# based on the assumption that the codes are accessible locally.

# If you plan to do remote execution, you need to first create
# a computer node via `verdi computer setup`,
# and then change the computer_name variable below accordingly.

computer_name=localhost
code_name=orca

# This needs to match the path where you mounted the code! 
ORCA_PATH=/opt/orca
export PATH=$ORCA_PATH:$PATH

verdi code show ${code_name}@${computer_name} 2>/dev/null ||      \
    verdi code setup                                              \
    --non-interactive                                             \
    --label ${code_name}                                          \
    --description "${code_name} code mounted via docker volume."  \
    --input-plugin ${code_name}_main                              \
    --computer ${computer_name}                                   \
    --remote-abs-path `which ${code_name}`                        \
    --prepend-text "export PATH=$ORCA_PATH:\$PATH"
