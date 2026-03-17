import os
import re
import argparse


def sort_and_update(directory, digits=None, dry_run=False):
	"""
	Zero-pad snapshot indices in .h5/.xdmf filenames and update .xdmf HDF5 references.

	Example:
	  3d_NACA0012_Re50000_AoA12_2D_6350000.h5
		-> 3d_NACA0012_Re50000_AoA12_2D_06350000.h5
	"""
	pattern = re.compile(r'^(3d_NACA0012_Re50000_AoA12_2D)_(\d+)\.(h5|xdmf)$')

	files = sorted(os.listdir(directory))
	pairs = {}

	for fname in files:
		match = pattern.match(fname)
		if not match:
			continue

		base = match.group(1)
		iteration = match.group(2)
		ext = match.group(3)

		pairs.setdefault(iteration, {'base': base})[ext] = fname

	if not pairs:
		print('No matching files found.')
		return

	max_len = max(len(iteration) for iteration in pairs)
	pad_digits = digits if digits is not None else max_len

	if pad_digits < max_len:
		raise ValueError(
			f'--digits ({pad_digits}) is smaller than current maximum ({max_len}).'
		)

	print(f'Found {len(pairs)} snapshot(s).')
	print(f'Padding width: {pad_digits}')

	for iteration in sorted(pairs.keys(), key=int):
		data = pairs[iteration]
		base = data['base']
		padded_iteration = iteration.zfill(pad_digits)

		old_h5_ref = f'{base}_{iteration}.h5'
		new_h5_ref = f'{base}_{padded_iteration}.h5'

		for ext in ('h5', 'xdmf'):
			old_name = data.get(ext)
			if old_name is None:
				continue

			new_name = f'{base}_{padded_iteration}.{ext}'
			old_path = os.path.join(directory, old_name)
			new_path = os.path.join(directory, new_name)

			if old_name == new_name:
				print(f'  Keep:   {old_name}')
				continue

			print(f'  Rename: {old_name}  ->  {new_name}')

			if not dry_run:
				if os.path.exists(new_path):
					raise FileExistsError(f'Target already exists: {new_path}')
				os.rename(old_path, new_path)

		old_xdmf_name = data.get('xdmf')
		if old_xdmf_name is None:
			continue

		new_xdmf_name = f'{base}_{padded_iteration}.xdmf'
		if dry_run:
			xdmf_read_path = os.path.join(directory, old_xdmf_name)
		else:
			xdmf_read_path = os.path.join(directory, new_xdmf_name)

		xdmf_write_path = os.path.join(directory, new_xdmf_name)

		with open(xdmf_read_path, 'r') as file:
			content = file.read()

		new_content = content.replace(old_h5_ref, new_h5_ref)

		if new_content != content:
			print(f'  Update xdmf: {old_h5_ref}  ->  {new_h5_ref}')
			if not dry_run:
				with open(xdmf_write_path, 'w') as file:
					file.write(new_content)
		else:
			print('  xdmf already up to date or reference not found.')

	if dry_run:
		print('\nDry-run complete. No files were modified.')
	else:
		print('\nDone.')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Zero-pad xy-slice snapshot indices and update xdmf HDF5 references.'
	)
	parser.add_argument(
		'directory',
		nargs='?',
		default='/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/xy_slices/',
		help='Directory containing the snapshot files.'
	)
	parser.add_argument(
		'--digits',
		type=int,
		default=None,
		help='Target index width (default: longest existing index length).'
	)
	parser.add_argument(
		'--dry-run',
		action='store_true',
		help='Preview changes without renaming or modifying any files.'
	)
	args = parser.parse_args()

	sort_and_update(args.directory, digits=args.digits, dry_run=args.dry_run)
