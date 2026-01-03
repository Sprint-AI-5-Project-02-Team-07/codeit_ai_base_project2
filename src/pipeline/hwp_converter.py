from glob import glob
import os
try:
    import win32com.client as win32
except ImportError:
    win32 = None
from pathlib import Path

def run_hwp_conversion(input_dir: str):
    """
    Convert all .hwp files in input_dir to .pdf
    """
    if win32 is None:
        print("[Error] win32com is not available. This feature works only on Windows.")
        return

    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"[Error] Input directory not found: {input_dir}")
        return

    hwp_files = list(input_path.glob('*.hwp'))
    if not hwp_files:
        print(f"[HWP Converter] No .hwp files found in {input_dir}")
        return

    print(f"[HWP Converter] Found {len(hwp_files)} HWP files. Starting conversion...")
    print("-------------------------NOTICE-------------------------")
    print("If a popup appears asking 'Try to access...', please select [Allow All].")
    print("--------------------------------------------------------")

    try:
        hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")
        hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule") # 보안 모듈 우회 시도
    except Exception as e:
        print(f"[Error] Failed to initialize HWP: {e}")
        return

    for hwp_file in hwp_files:
        pdf_file = hwp_file.with_suffix('.pdf')
        
        # Skip if PDF already exists
        if pdf_file.exists():
            print(f"⏩ Skipping (PDF exists): {hwp_file.name}")
            continue

        try:
            hwp.Open(str(hwp_file))
            # SaveAs format "PDF"
            hwp.SaveAs(str(pdf_file), "PDF")
            print(f"✅ Converted: {hwp_file.name} -> {pdf_file.name}")
        except Exception as e:
            print(f"❌ Failed to convert {hwp_file.name}: {e}")

    hwp.Quit()
    print("[HWP Converter] Conversion finished.")
