# EEGUnity Release Harness

AI-readable checklist for every PyPI release. Work through each section in order before uploading.

---

## 1. Version

- [ ] `eegunity/_version.py` — `__version__` matches the intended release tag.
- [ ] The same version does **not** already exist on PyPI (re-uploading is rejected).

---

## 2. Packaging integrity (run script)

```powershell
& "C:\Users\Cheng\anaconda3\envs\eegunity_release\python.exe" CI/pre_release_check.py
```

This script verifies:
- Every subpackage directory under `eegunity/` has an `__init__.py` (missing `__init__.py` silently drops the package from the wheel — this caused the 0.8.1 `kernel` incident).
- `setuptools.find_packages()` discovers all expected packages.
- The built wheel contains every discovered package.
- `twine check` passes.

**Do not upload if this script reports any errors.**

---

## 3. Build (clean rebuild)

```powershell
# Remove old artifacts
Get-ChildItem "dist\" | Remove-Item -Force
Remove-Item "build\" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "eegunity.egg-info\" -Recurse -Force -ErrorAction SilentlyContinue

# Build
& "C:\Users\Cheng\anaconda3\envs\eegunity_release\python.exe" -m build
```

---

## 4. CI tests

```powershell
& "C:\Users\Cheng\anaconda3\envs\eegunity_release\python.exe" CI/run_all_CI_test.py
```

All scripts must pass. CI config: `CI/CI_config.json` (data root `H:\`, output `E:\tmp\EEGUnity_CI\`).

---

## 5. Upload

```powershell
$env:PYTHONUTF8 = "1"
& "C:\Users\Cheng\anaconda3\envs\eegunity_release\python.exe" -m twine upload --disable-progress-bar dist/*
```

Credentials are read from `~/.pypirc`. On Windows the `--disable-progress-bar` flag is required to avoid a GBK encoding crash in twine's rich progress renderer.

---

## Known pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Subpackage missing `__init__.py` | `ModuleNotFoundError` after install; package absent from wheel | Add `__init__.py`; re-run step 2 |
| Re-uploading same version | `400 Bad Request` from PyPI | Yank broken release; bump version |
| Wrong Python environment | `ModuleNotFoundError: No module named 'mne'` | Use `eegunity_release` conda env |
| Windows GBK crash in twine | `UnicodeEncodeError: 'gbk'` | Add `--disable-progress-bar` and set `PYTHONUTF8=1` |
| `.pypirc` written with BOM | `InvalidConfiguration: Malformed configuration` | Re-write file without BOM (use the Write tool, not PowerShell `Out-File`) |
