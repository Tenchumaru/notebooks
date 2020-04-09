import pickle
from pathlib import Path
from utilities import pickle_iter

root_dir = Path(fr'D:\Dota 2\Heroes\Pickles')
target_dir = root_dir / 'Narrow'
target_dir.mkdir(exist_ok=True)
for input_path in root_dir.glob('*.pickle'):
    print(input_path.name)
    output_path = target_dir / input_path.name
    with open(str(output_path), 'wb') as fout:
        for image in pickle_iter(str(input_path)):
            pickle.dump(image[:, 44:-44], fout)
