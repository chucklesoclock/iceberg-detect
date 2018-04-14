print('start'.center(20, '-'))
with open('predict-is_iceberg.ipynb') as f:
    source = []
    num_bracket = 0
    in_source = False
    for lnum, line in enumerate(f):
        # print('{:3d}: {}'.format(lnum, line.strip()))
        lnum += 1
        sline = line.strip()
        prefix = '"source": ['
        if sline.startswith(prefix):
            num_bracket = 1
            in_source = True
            continue
        elif in_source:
            if '[' in sline:
                num_bracket += 1
            if ']' in sline:
                num_bracket -= 1
        if num_bracket == 0:
            in_source = False
        else:
            source.append(sline)

    out_file = '\n'.join(l.strip('"').rstrip('\n",').replace('\\n', '')
                         for l in source)

with open('predict-is_iceberg-lost_source.txt', 'w') as out:
    out.write(out_file)

print('end'.center(20, '-'))
