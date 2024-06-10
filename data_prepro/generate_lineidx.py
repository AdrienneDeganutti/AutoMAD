def generate_lineidx_files(input_file, output_prefix):
    line_offsets = []
    with open(input_file, 'rb') as f:
        offset = 0
        while True:
            line = f.readline()
            if not line:
                break
            line_offsets.append(offset)
            offset = f.tell()

    # Write the .lineidx file
    with open(output_prefix + '.lineidx', 'w') as f:
        for offset in line_offsets:
            f.write(str(offset) + '\n')

    # Write the .lineidx.8b file
    with open(output_prefix + '.lineidx.8b', 'wb') as f:
        for offset in line_offsets:
            f.write(offset.to_bytes(8, 'little'))

if __name__ == '__main__':
    input_file = 'datasets/32_clips_tsv/32_clips_train_features.tsv'      # SET INPUT HERE
    output_prefix = 'datasets/32_clips_tsv/32_clips_train_features'       # SET OUTPUT HERE
    generate_lineidx_files(input_file, output_prefix)
    print(f"Generated {output_prefix}.lineidx and {output_prefix}.lineidx.8b files.")