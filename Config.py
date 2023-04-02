import os
# ==================================================      ==================================================
class Config () :
    buildUsers = ['vicying']
    productNames = ['Example']

os.environ["POSTGRES_OPSNABAGEMENT_DATABASE"] = "scientificanalysis"
os.environ["POSTGRES_OPSNABAGEMENT_SCHEMA"] = "public"
os.environ["STORAGE_RECORDSAVEPATH"] = "mfs/OPSData"

import LocalConfig



