import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from glasspy.predict import VITRIFY

def main():
    inp,out=map(Path,sys.argv[1:3])
    comp=pd.read_csv(inp)
    model=VITRIFY()
    p=np.asarray(model.predict_proba_glass(comp),dtype=float).reshape(-1)
    out.write_text(json.dumps([float(x) for x in p]),encoding='utf-8')
if __name__=='__main__':main()
