# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


from PyInstaller.utils.hooks import collect_submodules, collect_data_files
tf_hidden_imports = collect_submodules('tensorflow_core')
tf_datas = collect_data_files('tensorflow_core', subdir=None, include_py_files=True)

a = Analysis(['AvpSongPulseAnnotations.py'],
             pathex=['G:\\AVP\\App'],
             binaries=[('C:\\Users\\permo\\Anaconda3\\Library\\plugins\\platforms\\*.*', '.')],
             datas=tf_datas,
             hiddenimports=tf_hidden_imports +['pkg_resources.py2_warn', 'sklearn.utils._cython_blas', 'sklearn.neighbors._typedefs', 'sklearn.neighbors._quad_tree', 'sklearn.tree', 'sklearn.tree._utils'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='AvpSongPulseAnnotations',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          icon='G:\\AVP\\APP\\window_oscillograph.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='AvpSongPulseAnnotations')
