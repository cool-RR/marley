#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}"
pytest -v --showlocals --tb=long "$@" || true

temp_pytest_report=$(mktemp)
python misc/pytest_stuff/process_html_report.py pytest_report.html > "${temp_pytest_report}"
rm pytest_report.html
mv "${temp_pytest_report}" pytest_report.html

