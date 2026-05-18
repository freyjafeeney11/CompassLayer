import os
import subprocess
import sys

def build():
    print('Checking for PyInstaller...')
    try:
        import PyInstaller
    except ImportError:
        print('Installing PyInstaller...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])
    try:
        import pyo
    except ImportError:
        print("Error: 'pyo' is not installed. Please pip install it before building.")
        sys.exit(1)
    pyo_path = os.path.dirname(pyo.__file__)
    print(f'Found pyo at: {pyo_path}')
    entry_point = 'run_live.py'
    sep = os.pathsep
    data_args = ['--add-data', f'{pyo_path}{sep}pyo']
    assets_dir = 'assets'
    if os.path.exists(assets_dir):
        print('Scanning assets folder...')
        for item in os.listdir(assets_dir):
            if item.lower() == 'videos':
                print(" -> Skipping 'videos' folder to keep EXE size small.")
                continue
            source_path = os.path.join(assets_dir, item)
            if os.path.isdir(source_path):
                dest_path = f'assets/{item}'
            else:
                dest_path = 'assets'
            data_args.extend(['--add-data', f'{source_path}{sep}{dest_path}'])
            print(f' -> Adding: {item}')
    else:
        print(f"Warning: Could not find an '{assets_dir}' directory.")
    hidden_imports = ['--hidden-import', 'pyo', '--hidden-import', 'win32com', '--hidden-import', 'win32com.client', '--hidden-import', 'pythoncom', '--hidden-import', 'pyttsx3', '--hidden-import', 'mss', '--hidden-import', 'keyboard', '--hidden-import', 'cv2']
    cmd = [sys.executable, '-m', 'PyInstaller', '--onefile', '--name', 'CompassLayer', '--clean'] + data_args + hidden_imports + [entry_point]
    print('\nStarting PyInstaller...')
    subprocess.check_call(cmd)
    print('\n' + '=' * 40)
    print('BUILD COMPLETE!')
    print(f"Your EXE is in the 'dist' folder.")
    print("Remember to place your 'videos' folder next to the finished EXE!")
    print('=' * 40)
if __name__ == '__main__':
    build()