"""Train this project's TensorFlow model using explicit data splits."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from portfolio_core.train import main
if __name__=='__main__':
    sys.argv.insert(1,'sign-language-translator')
    main()
