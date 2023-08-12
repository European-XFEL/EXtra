###########################################################################################
#                                                                                         #
# Note: if this script is updated make sure to update the copy in xsoft's home directory! #
#                                                                                         #
###########################################################################################

set -euxo pipefail

function check_and_install()
{
    local repo_url='git+https://github.com/European-XFEL/EXtra.git'
    # Command to do a dry-run install
    local pip_cmd="pip install ${repo_url} --dry-run"

    # For some unfathomable reason getting the return code from a subshell
    # doesn't work unless a local variable is declared *before* assignment.
    local pip_out
    pip_out=$(eval 2>&1 ${pip_cmd})

    # If the command itself failed, then exit
    if [[ ! $? -eq 0 ]]; then
        echo "Pip command \`${pip_cmd}\` failed with output:"
        echo "${pip_out}"
        return 1
    fi

    # If nothing would be installed at all, including a newer version of the
    # package, then exit early.
    if [[ ! ${pip_out} == *'Would install'* ]]; then
        version=$(pip freeze | grep 'euxfel-EXtra')
        echo "Nothing to do, installed version is already up-to-date: ${version}"
        return 0
    fi

    # Pip prints the packages it'd install on the last line, so we strip the
    # prefix and count the number of packages that would be installed.
    local pkgs_str=$(echo "${pip_out}" | tail -1 | sed 's/Would install //')
    local num_new_pkgs=$(echo ${pkgs_str} | wc -w)

    # If more than one package would be installed, i.e. a new version and
    # dependencies, then we error out.
    if [[ ${num_new_pkgs} -gt 1 ]]; then
        echo 'Error: pip would install new packages! Dependencies need to be installed or updated with conda.'
        echo
        echo "Output from \`${pip_cmd}\`:"
        echo "${pip_out}"

        return 1
    fi

    # Otherwise, we can safely install the latest version
    pip install ${repo_url} --no-deps
}

# Always install into the current environment
source /usr/share/Modules/init/bash
module load exfel exfel-python

check_and_install
