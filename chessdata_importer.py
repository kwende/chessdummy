import fire
import zstandard

# https://python-zstandard.readthedocs.io/en/latest/decompressor.html

def import_data(lichess_data_file: str, output_path: str) -> None:

    # decompress the data from the standard .zstd format used by lichess
    decompressor = zstandard.ZstdDecompressor()
    with open(lichess_data_file, 'rb') as input, open(output_path, 'wb') as output:
        decompressor.copy_stream(input, output)

if __name__ == '__main__':
    fire.Fire()