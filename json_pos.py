import json

print('start'.center(20, '-'))
with open('predict-is_iceberg.ipynb') as f:
    # for line in f:
    #     print(line.rstrip())
    out_file = ''.join((line.rstrip() for line in f))
print('end'.center(20, '-'))
json.loads(out_file)