import os
import re
import argparse

def rename_and_update(directory, dry_run=False):
    """
    Rename snapshot pairs and update xdmf content.

    Old names: 3d_NACA0012_Re50000_AoA12_XXXXXXXX-2D-Z65.h5 / .xdmf
    New names: 3d_NACA0012_Re50000_AoA12_2D_XXXXXXXX.h5 / .xdmf
    """
    pattern = re.compile(r'^(3d_NACA0012_Re50000_AoA12)_(\d+)-2D-Z65\.(h5|xdmf)$')

    files = sorted(os.listdir(directory))
    pairs = {}

    for fname in files:
        m = pattern.match(fname)
        if m:
            iteration = m.group(2)
            ext = m.group(3)
            pairs.setdefault(iteration, {})[ext] = fname

    if not pairs:
        print('No matching files found.')
        return

    print(f'Found {len(pairs)} snapshot(s).')

    for iteration, exts in sorted(pairs.items()):
        for ext, old_name in exts.items():
            new_name = f'3d_NACA0012_Re50000_AoA12_2D_{iteration}.{ext}'
            old_path = os.path.join(directory, old_name)
            new_path = os.path.join(directory, new_name)

            print(f'  Rename: {old_name}  ->  {new_name}')

            if not dry_run:
                os.rename(old_path, new_path)

        # Update xdmf content after renaming
        xdmf_new_name = f'3d_NACA0012_Re50000_AoA12_2D_{iteration}.xdmf'
        xdmf_path = os.path.join(directory, xdmf_new_name)

        old_h5_ref = f'3d_NACA0012_Re50000_AoA12_{iteration}-2D-Z65.h5'
        new_h5_ref = f'3d_NACA0012_Re50000_AoA12_2D_{iteration}.h5'

        if dry_run:
            # In dry-run mode read from old path since file was not renamed
            xdmf_read_path = os.path.join(directory, f'3d_NACA0012_Re50000_AoA12_{iteration}-2D-Z65.xdmf')
        else:
            xdmf_read_path = xdmf_path

        with open(xdmf_read_path, 'r') as f:
            content = f.read()

        new_content = content.replace(old_h5_ref, new_h5_ref)

        if new_content != content:
            print(f'  Update xdmf: {old_h5_ref}  ->  {new_h5_ref}')
            if not dry_run:
                with open(xdmf_path, 'w') as f:
                    f.write(new_content)
        else:
            print(f'  xdmf already up to date or reference not found.')

    if dry_run:
        print('\nDry-run complete. No files were modified.')
    else:
        print('\nDone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Rename xy-slice snapshots and update xdmf HDF5 references.'
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/xy_slices',
        help='Directory containing the snapshot files (default: xy_slices directory).'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without renaming or modifying any files.'
    )
    args = parser.parse_args()

    rename_and_update(args.directory, dry_run=args.dry_run)
