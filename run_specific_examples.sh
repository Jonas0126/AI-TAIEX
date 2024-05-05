#!/bin/bash
usage() {
    echo "Usage: $0 [-t <string>] [-m <string>] [-y <string>]" 1>&2
    exit 1;
}

usage_help() {
    echo "Usage: $0 [-t <string>] [-m <string>] [-y <string>]"
    echo "Description: "
    echo "  -t    target"
    echo "  -m    model"
    echo "  -y    year"
    exit 0;
}

all_targets=("Taiex" "SP500")
all_models=("LR" "XGB" "LSTM" "CRNN" "TCN")
all_years=("2018" "2019" "2020" "2021" "2022" "2023")

usage_target() {
    echo "Option Error: -t value should be ${all_targets[*]}" 1>&2
    exit 1;
}

usage_model() {
    echo "Option Error: -m value should be ${all_models[*]}" 1>&2
    exit 1;
}

usage_year() {
    echo "Option Error: -y value should be ${all_years[*]}" 1>&2
    exit 1;
}

while getopts ":t:m:y:h" opt; do
    case "${opt}" in
        t)
            for item in ${all_targets[*]}; do
                if [[ "${item}" == "${OPTARG}" ]]; then
                    target=${OPTARG}
                    break
                fi
            done
            if [ -z "${target}" ]; then
                usage_target
            fi
            ;;
        m)
            for item in ${all_models[*]}; do
                if [ "${item}" = "${OPTARG}" ]; then
                    model=${OPTARG}
                    break
                fi
            done
            if [ -z "${model}" ]; then
                usage_model
            fi
            ;;
        y)
            for item in ${all_years[*]}; do
                if [ "${item}" = "${OPTARG}" ]; then
                    year=${OPTARG}
                    break
                fi
            done
            if [ -z "${year}" ]; then
                usage_year
            fi
            ;;
        h)
            usage_help
            ;;
        *)
            usage
            ;;
    esac
done

shift "$((OPTIND-1))"

if [ -z "${target}" ]; then
    usage
fi

echo "Target: ${target}"
echo "Model: ${model}"
echo "Year: ${year}"

for filename in ./example/${target}/*${model}*${year}*.yaml; do
    [ ! -f $filename ] && continue
    echo "python main.py $filename"
    python3 main.py $filename
done
for filename in ./example/${target}/*${model}*${year}*.yaml; do
    [ ! -f $filename ] && continue
    if [[ "$filename" != *"LR"* ]] && [[ "$filename" != *"XGB"* ]]; then
        echo "python finetune_when_trade.py $filename"
        python3 finetune_when_trade.py $filename
    fi
done