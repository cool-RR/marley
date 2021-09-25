#!python

import fileinput

for line in fileinput.input():
    print(line.replace('td class="col-result', 'td class="col-result fake-button-for-vimium'))
