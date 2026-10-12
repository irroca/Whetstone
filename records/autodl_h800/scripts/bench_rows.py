import json
import sys

for path in sys.argv[1:]:
    for row in json.load(open(path))["rows"]:
        print(f"{path.rsplit('/', 1)[-1]:<20} {row.get('attention', 'default'):<8} bs{row['batch_size']:<3} "
              f"{row['tokens_per_s']:>9,.0f} tok/s  {row['peak_memory_gb']:.1f} GB  MFU {row['mfu']:.1%}")
