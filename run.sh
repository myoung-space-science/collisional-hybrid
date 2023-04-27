#!/bin/bash

# Exit immediately if a pipeline returns non-zero status. See
# https://www.gnu.org/software/bash/manual/html_node/The-Set-Builtin.html
set -e

# Declare the name of the target program.
prog=hybrid

# Declare project-related directories.
rootdir=/home/matthew/sandbox/dmswarm-hybrid
rundir=${rootdir}/runs
srcdir=${rootdir}/src
bindir=${rootdir}/bin

# Set option defaults.
verbose=0
np=2
debug=0
outdir=
outname=vectors
outtype=hdf
extra=

# Define text formatting commands.
# - textbf: Use bold-face.
# - textnm: Reset to normal.
# - startul: Start underlined text.
# - endul: End underlined text.
textbf=$(tput bold)
textnm=$(tput sgr0)
startul=$(tput smul)
endul=$(tput rmul)

# This is the CLI's main help text.
show_help()
{
    cli_name=${0##*/}
    echo "
${textbf}NAME${textnm}
        $cli_name - Run the hybrid simulation

${textbf}SYNOPSIS${textnm}
        ${textbf}$cli_name${textnm} [${startul}OPTIONS${endul}]

${textbf}DESCRIPTION${textnm}
        This script will make and execute ${textbf}${prog}${textnm}.
        
        The following options pertain to building and running the executable, 
        and saving the output. All additional options will pass through to 
        ${textbf}${prog}${textnm}.

        ${textbf}-h${textnm}, ${textbf}--help${textnm}
                Display help and exit.
        ${textbf}-n${textnm}, ${textbf}--nproc${textnm} ${startul}N${endul}
                Number of processors to use (default: ${np}).
        ${textbf}-o${textnm}, ${textbf}--outdir${textnm} ${startul}DIR${endul}
                Name of the output directory within ${rundir}.
                The default name is generated from the current date and time.
        ${textbf}--outname${textnm} ${startul}F${endul}
                Name of the file, without extension, that will contain vectors 
                of simulated quantities.
                (default: '${outname}')
        ${textbf}--debug${textnm}
                Start in the debugger (default: false).
        ${textbf}-v${textnm}, ${textbf}--verbose${textnm}
                Print runtime messages (default: false).
"
}

# This will run if the CLI gets an unrecognized option.
report_bad_arg()
{
    printf "\nUnrecognized command: ${1}\n\n"
}

# Read command-line arguments.
while [ "${1}" != "" ]; do
    case "${1}" in
        -n | --nproc )
            shift
            np="${1}"
            ;;
        -o | --outdir )
            shift
            outdir="${1}"
            ;;
        --outname )
            shift
            outname="${1}"
            ;;
        -v | --verbose )
            verbose=1
            ;;
        --debug )
            debug=1
            ;;
        -h | --help )
            show_help
            exit
            ;;
        * )
            extra="${extra} ${1}"
            ;;
    esac
    shift
done

stage=

cleanup() {
    if [ "${stage}" != "Complete" ]; then
        exitmsg="Exiting."
        if [ -n "${stage}" ]; then
            exitmsg="${stage} stage failed. ${exitmsg}"
        fi
        echo $exitmsg
        exit 1
    fi
}

trap cleanup EXIT

# Mark this stage.
stage="Setup"

# Set the output file name.
outfile="${outname}.${outtype}"

# Set the local name of the output directory.
if [ -z "${outdir}" ]; then
    outdir=$(date +'%Y-%m-%d-%H%M%S')
else
    outdir=$(readlink -f "${outdir}")
fi

# Create the full output directory.
dstdir=${rundir}/${outdir}
mkdir -p ${dstdir}

# Mark this stage.
stage="Symlink"

# Create a symlink to this run in the directory of runs.
cd ${rundir}
rm -f latest
ln -s ${outdir} latest

if [ ${verbose} == 1 ]; then
    vflag="-v"
fi

# Mark this stage.
stage="Build"

# Build the executable in the source directory.
cd ${srcdir}
make ${prog} &> ${dstdir}/build.log

# Move to the output directory.
cd ${dstdir}

# Mark this stage.
stage="Run"

# Run the program.
if [ ${debug} == 1 ]; then
    mpiexec -n ${np} ${bindir}/${prog} \
        -debug_terminal "gnome-terminal --" \
        -start_in_debugger \
        --outname ${outname} \
        ${extra}
else
    mpiexec -n ${np} ${bindir}/${prog} \
        -ksp_monitor_short \
        --outname ${outname} \
        ${extra} &> run.log
fi

# Signal success to the clean-up function.
stage="Complete"

