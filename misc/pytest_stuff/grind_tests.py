#!python

import subprocess
import pathlib
import sys
import datetime as datetime_module

repo_root: pathlib.Path = pathlib.Path(__file__).parent.parent.parent.resolve()
assert repo_root.is_dir()
assert repo_root.name in ('grid_royale', 'Marley')

pytest_sh_path = (repo_root / 'pytest.sh')
assert pytest_sh_path.exists()

time_string = datetime_module.datetime.now().isoformat().replace(':', '.')
pytest_reports_folder: pathlib.Path = (repo_root / 'pytest_reports' / time_string)
pytest_reports_folder.mkdir(exist_ok=True, parents=True)

pytest_report_path = (repo_root / 'pytest_report.html')


for i in range(100):
    subprocess.run((r'C:\Program Files\Git\usr\bin\bash.exe', str(pytest_sh_path),
                    *sys.argv[1:]), check=False)
    pytest_report_path: pathlib.Path
    pytest_report_path.rename(pytest_reports_folder / f'pytest_report_{i:04d}.html')


